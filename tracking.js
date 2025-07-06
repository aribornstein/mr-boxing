// tracking.js
import * as THREE from 'three';

const coco_labels = [
  'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant',
  'stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
  'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
  'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
  'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
  'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
  'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
];

export async function enumerateCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter(d => d.kind === "videoinput");
}

export async function startStereoCameras(videoLeftElem, videoRightElem, leftId, rightId) {
  const streamLeft = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: leftId } } });
  const streamRight = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: rightId } } });
  videoLeftElem.srcObject = streamLeft;
  videoRightElem.srcObject = streamRight;
  return [streamLeft, streamRight];
}

function preprocess(img, size=640) {
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');
  const [iw, ih] = [img.naturalWidth || img.width, img.naturalHeight || img.height];
  const r = Math.min(size / iw, size / ih);
  const newW = Math.round(iw * r), newH = Math.round(ih * r);
  const dx = Math.floor((size - newW) / 2), dy = Math.floor((size - newH) / 2);
  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(img, dx, dy, newW, newH);

  const imageData = ctx.getImageData(0, 0, size, size).data;
  const floatArray = new Float32Array(size * size * 3);
  for (let i = 0; i < size * size; ++i) {
    floatArray[i] = imageData[i * 4] / 255.0;
    floatArray[size * size + i] = imageData[i * 4 + 1] / 255.0;
    floatArray[2 * size * size + i] = imageData[i * 4 + 2] / 255.0;
  }
  return new window.ort.Tensor('float32', floatArray, [1, 3, size, size]);
}

async function loadYOLOModel(onnxURL) {
  if (!window.ort) {
    await import('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');
  }
  const resp = await fetch(onnxURL);
  const buffer = await resp.arrayBuffer();
  const session = await window.ort.InferenceSession.create(buffer, { executionProviders: ['wasm'] });
  return session;
}

async function runYOLO(session, img) {
  const tensor = preprocess(img, 640);
  const feeds = {}; feeds[session.inputNames[0]] = tensor;
  const results = await session.run(feeds);
  let output = results[session.outputNames[0]];
  let dets = output.data;
  let detections = [];
  for (let i = 0; i < dets.length; i += 6) {
    const x1 = dets[i], y1 = dets[i+1], x2 = dets[i+2], y2 = dets[i+3], score = dets[i+4], class_id = dets[i+5];
    if (score < 0.7) continue;
    detections.push({
      box: [x1, y1, x2, y2],
      score, class: class_id,
      label: coco_labels[class_id] || `class${class_id}`
    });
  }
  return detections;
}

function getCenter(box) {
  return [ (box[0]+box[2])/2, (box[1]+box[3])/2 ];
}

function matchDetections(left, right, yThreshold=30) {
  let matches = [];
  right = [...right];
  left.forEach(lbox => {
    let lcy = getCenter(lbox.box)[1];
    let candidates = right.filter(r => r.label === lbox.label);
    if (!candidates.length) return;
    let best = candidates.reduce((a,b) => {
      let bcy = getCenter(b.box)[1];
      return (Math.abs(bcy - lcy) < Math.abs(getCenter(a.box)[1] - lcy)) ? b : a;
    });
    let bcy = getCenter(best.box)[1];
    if (Math.abs(bcy - lcy) < yThreshold) {
      matches.push({label: lbox.label, left: lbox, right: best});
      right.splice(right.indexOf(best), 1);
    }
  });
  return matches;
}

function triangulate(uL, uR, vL, fx=600, fy=600, cx=320, cy=240, baseline=0.065) {
  const disparity = uL - uR;
  if (!disparity) return null;
  const Z = (fx * baseline) / disparity;
  const X = (uL - cx) * Z / fx;
  const Y = (vL - cy) * Z / fy;
  return {x: X, y: Y, z: -Z};
}

// ---- Particle Helper ----
function createParticleGroup(color = 0x00fffc, size = 0.18) {
  const group = new THREE.Group();
  const textureLoader = new THREE.TextureLoader();
  const spriteMap = textureLoader.load('https://cdn.jsdelivr.net/gh/mrdoob/three.js@r150/examples/textures/sprites/spark1.png');
  for (let i = 0; i < 8; ++i) {
    const material = new THREE.SpriteMaterial({ map: spriteMap, color, transparent: true, opacity: 0.77 });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(size, size, size);
    const phi = Math.random() * Math.PI * 2;
    const theta = Math.random() * Math.PI;
    sprite.position.set(
      Math.sin(theta) * Math.cos(phi) * (size * 0.7),
      Math.sin(theta) * Math.sin(phi) * (size * 0.7),
      Math.cos(theta) * (size * 0.7)
    );
    group.add(sprite);
  }
  return group;
}

/**
 * Main entry point for YOLO + stereo tracking + collider + particle visual integration.
 * @param {object} options - see below.
 * @returns {object} detectedObjectColliders - always up-to-date with tracked objects.
 */
export function setupTracking({
  physicsWorld,
  RAPIER,
  scene,               // <--- YOUR THREE.Scene instance!
  videoLeftElem,
  videoRightElem,
  onnxURL = 'yolo12nms.onnx',
  minScore=0.7,
  matchYThreshold=30,
  fx=600, fy=600, cx=320, cy=240, baseline=0.065
}) {
  const detectedObjectColliders = {}; // label => { body, collider, particleGroup }

  async function frameToImage(video) {
    const c = document.createElement('canvas');
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext('2d').drawImage(video, 0, 0);
    const img = new window.Image();
    img.src = c.toDataURL();
    return new Promise(res=>img.onload=()=>res(img));
  }

  async function trackingLoop(yoloSession) {
    try {
      const [imgL, imgR] = await Promise.all([
        frameToImage(videoLeftElem),
        frameToImage(videoRightElem)
      ]);
      const [detsL, detsR] = await Promise.all([
        runYOLO(yoloSession, imgL),
        runYOLO(yoloSession, imgR)
      ]);
      const matches = matchDetections(detsL, detsR, matchYThreshold);

      // Track current frame's active labels
      const currentLabels = new Set();

      matches.forEach(({label, left, right}) => {
        currentLabels.add(label);
        const [uL,vL] = getCenter(left.box);
        const [uR,vR] = getCenter(right.box);
        const pos = triangulate(uL, uR, vL, fx, fy, cx, cy, baseline);
        if (!pos) return;
        const boxPx = left.box[2]-left.box[0], boxPy = left.box[3]-left.box[1];
        const size = {x: boxPx*Math.abs(pos.z)/fx, y: boxPy*Math.abs(pos.z)/fy, z: 0.08};

        let obj = detectedObjectColliders[label];
        if (!obj) {
          const body = physicsWorld.createRigidBody(RAPIER.RigidBodyDesc.kinematicPositionBased());
          const collider = physicsWorld.createCollider(
            RAPIER.ColliderDesc.cuboid(size.x/2, size.y/2, size.z/2).setSensor(true),
            body
          );
          // --- Add Particle Effect! ---
          let color = 0x00fffc;
          if (label === 'sports ball') color = 0xffff00;
          else if (label === 'person') color = 0x3399ff;
          const particleGroup = createParticleGroup(color, Math.max(size.x, size.y, 0.12));
          scene.add(particleGroup);
          detectedObjectColliders[label] = { body, collider, particleGroup };
        }
        // Update physics and visual position every frame
        detectedObjectColliders[label].body.setNextKinematicTranslation(pos);
        if (detectedObjectColliders[label].particleGroup) {
          detectedObjectColliders[label].particleGroup.position.set(pos.x, pos.y, pos.z);
          detectedObjectColliders[label].particleGroup.visible = true;
        }
      });

      // --- CLEANUP: remove colliders and particles no longer detected ---
      Object.keys(detectedObjectColliders).forEach(label => {
        if (!currentLabels.has(label)) {
          const { body, collider, particleGroup } = detectedObjectColliders[label];
          physicsWorld.removeCollider(collider, true);
          physicsWorld.removeRigidBody(body);
          if (particleGroup && scene) scene.remove(particleGroup);
          delete detectedObjectColliders[label];
        }
      });

      setTimeout(() => trackingLoop(yoloSession), 150);
    } catch (e) {
      console.error("Tracking error:", e);
      setTimeout(() => trackingLoop(yoloSession), 500);
    }
  }

  // Load the YOLO model and start the loop!
  loadYOLOModel(onnxURL).then(yoloSession => {
    trackingLoop(yoloSession);
  });

  // Expose colliders so you can check for collisions in main code:
  return detectedObjectColliders;
}
