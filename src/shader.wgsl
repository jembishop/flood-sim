
struct Point {
  u: f32,
  z: f32,
  f: f32,
  kappa: f32,
  grad_u: vec2<f32>,
  div_term: f32,
  pad: f32,
}


const dt: f32 = 0.00002;

const alpha_m: f32 = 1.6666667;
const c_f: f32 = 1.;
const gamma_m: f32 = 0.5;


const dims_real_x: f32 = 1.;
const dims_real_y: f32 = 1.;

@group(0)
@binding(0)
var<storage, read_write> fluid: array<Point>;

@group(0)
@binding(1)
var<uniform> dims: vec2<u32>;


fn get_index(coords: vec2<u32>) -> u32 {
    return dims.x * coords.y + coords.x;
}

fn kappa(u: f32, grad_u: vec2<f32>, z: f32) -> f32 {
   let mag = length(grad_u);
   let zz = max(u - z, 0.);
   let num = pow(zz , alpha_m);
   let denom = (c_f * pow(mag, 1. - gamma_m) + 0.001);
   let d = num / denom;
   return d;
  //  return  0.;
}

fn get_p(coords: vec2<u32>, dir: vec2<i32>) -> Point {
    let c = vec2<i32>(coords);
    let cc = c + dir;
    if is_outside(cc) {
      // return fluid[get_index(coords)];
      return Point(0., 0., 0., 0., vec2<f32>(0., 0.), 0., 0.);
    }
    return fluid[get_index(vec2<u32>(cc))];
}

fn compute_grad_u(coords: vec2<u32>) -> vec2<f32> {
   let xm1 = get_p(coords, vec2(-1, 0)).u;
   let x1 = get_p(coords, vec2(1, 0)).u;
   let ym1 = get_p(coords, vec2(0, -1)).u;
   let y1 = get_p(coords, vec2(0, 1)).u;
   let dims_real = vec2(dims_real_x, dims_real_y);
   let ds = dims_real / vec2<f32>(dims);
   return vec2((x1 - xm1) / 2., (y1 - ym1) / 2.) / ds;
}

fn is_outside(coord: vec2<i32>) -> bool {
    if (coord.x < 0) || (coord.x >= i32(dims.x))
    || (coord.y < 0) || (coord.y >= i32(dims.y)) {
       return true;
    }
    return false;
}


fn div_kappa_grad_u(coords: vec2<u32>) -> f32 {
   let pxm1 = get_p(coords ,vec2(-1, 0));
   let xm1 = pxm1.grad_u.x * pxm1.kappa;
  //  let xm1 = pxm1.grad_u.x;

   let px1 = get_p(coords ,vec2(1, 0));
   let x1 = px1.grad_u.x * px1.kappa;
  //  let x1 = px1.grad_u.x;

   let pym1 = get_p(coords ,vec2(0, -1));
   let ym1 = pym1.grad_u.y * pym1.kappa;
  //  let ym1 = pym1.grad_u.y;

   let py1 = get_p(coords ,vec2(0, 1));
   let y1 = py1.grad_u.y * py1.kappa;
  //  let y1 = py1.grad_u.y;

   let dims_real = vec2(dims_real_x, dims_real_y);
   let ds = dims_real / vec2<f32>(dims);
   let dx = (x1 - xm1) / (2. * ds.x);
   let dy = (y1 - ym1) / (2. * ds.y);
   return dx + dy;
}

const N_ITERS_PER_FRAME: u32 = 128;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) grid: vec3<u32>) {
    for(var i: i32 = 0; i < N_ITERS_PER_FRAME; i++) {
      let coords = grid.xy;
      let idx = get_index(coords);
      let grad_u = compute_grad_u(coords);
      fluid[idx].grad_u = grad_u;
      let z = fluid[idx].z;
      let f = fluid[idx].f;
      let u = fluid[idx].u;
      fluid[idx].kappa = kappa(u, grad_u, z);
      storageBarrier();
      let div_term = div_kappa_grad_u(coords);
      fluid[idx].div_term = div_term;
      fluid[idx].u += dt * (div_term + f);
      storageBarrier();
    }
}