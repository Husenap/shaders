// Available uniforms
//#=============================#
//* uniform float iTime;        *
//* uniform vec2 iResolution;   *
//* uniform int iFrame;         *
//* uniform vec3 iCameraOrigin; *
//* uniform vec3 iCameraTarget; *
//#=============================#

#ifndef TAKOYAKI
const float FocalLength = 0.02;
const float CameraDistance = -1000.0;
const float HeartSize = 15.0;
const vec4 GroundAlbedo = vec4(0.823, 1.0, 0.918, 1.000);
const vec4 GroundAlbedo2 = vec4(0.960, 0.838, 1.000, 1.000);
const vec4 HeartAlbedo = vec4(0.962, 0.962, 0.962, 1.0);

const vec3 LightPositionLeft = vec3(-107.9, 99.9, 30.2);
const vec3 LightPositionRight = vec3(108.3, 94.8, 30.5);
const vec3 LightPositionBack = vec3(0.0, 152.5, 496.7);

const vec4 LightColorLeft = vec4(0.150, 0.466, 0.977, 1.589);
const vec4 LightColorRight = vec4(0.330, 0.208, 1.000, 1.534);
const vec4 LightColorBack = vec4(0.622, 0.523, 1.000, 0.561);

const vec2 LightDistanceDecayLeft = vec2(427.2, 1.0);
const vec2 LightDistanceDecayRight = vec2(392.7, 1.0);
const vec2 LightDistanceDecayBack = vec2(2934.0, 1.0);

const vec4 AmbientLight = vec4(0.233, 0.090, 0.900, 0.808);
#endif

//======================================================
// CONSTANTS
//======================================================
const float Epsilon = 1e-4;

const int RayMarchingSteps = 1024;
const float RayMarchingSurfaceDistance = 5e-4;
const float RayMarchingMaxDistance = 10000.0;
const float RayMarchingMaxBounces = 1.0;

const mat4 DitherPattern = mat4(0.0, 12., 3.0, 15., 8.0, 4.0, 11., 7.0, 2.0,
                                14., 1.0, 13., 10., 6.0, 9.0, 5.0);

const int IdGround = 0;
const int IdHeart = 1;

const float PI = 3.1415926535;

#define TIME (iTime)
#define ZERO (min(iFrame, 0))

//======================================================
// DEBUG
//======================================================
//#define DEBUG_ITERATIONS
//#define DEBUG_NORMALS
//#define DEBUG_ALBEDO
//#define DEBUG_ROUGHNESS
//#define DEBUG_METALLIC
//#define DEBUG_AO
//#define DEBUG_LOCAL_POSITION
//#define DEBUG_DIFFUSE
//#define DEBUG_SPECULAR

//======================================================
// Structs
//======================================================
struct Ray {
  vec3 origin;
  vec3 direction;
} _Ray;
struct Hit {
  float depth;
  int id;
  vec3 localPosition;
} _Hit;

struct Material {
  vec3 albedo;
  float roughness;
  float metallic;
} _Material;
struct GeometricContext {
  vec3 position;
  vec3 normal;
  vec3 viewDir;
} _GeometricContext;
struct IncidentLight {
  vec3 direction;
  vec3 color;
  bool visible;
} _IncidentLight;
struct ReflectedLight {
  vec3 directDiffuse;
  vec3 directSpecular;
  vec3 indirectDiffuse;
  vec3 indirectSpecular;
} _ReflectedLight;
struct PointLight {
  vec3 position;
  vec3 color;
  float visibleDistance;
  float decay;
} _PointLight;

//======================================================
// UTILS
//======================================================
float saturate(const in float x) { return clamp(x, 0.0, 1.0); }
vec3 saturate(const in vec3 x) { return clamp(x, vec3(0.0), vec3(1.0)); }
vec3 hash33(vec3 p) {
  p = vec3(dot(p, vec3(127.1, 311.7, 74.7)), dot(p, vec3(269.5, 183.3, 246.1)),
           dot(p, vec3(113.5, 271.9, 124.6)));

  return fract(sin(p) * 43758.5453123);
}
float map(const in float value, const in float low1, const in float high1,
          const in float low2, const in float high2) {
  return low2 + (value - low1) * (high2 - low2) / (high1 - low1);
}
vec2 hash22(in vec2 p) {
  vec3 a = fract(p.xyx * vec3(123.34, 234.34, 345.65));
  a += dot(a, a + 34.45);
  return fract(vec2(a.x * a.y, a.y * a.z));
}

float voronoi(vec2 uv) {
  vec2 gv = fract(uv) - 0.5;
  vec2 id = floor(uv);

  float minDist = 99.;
  vec2 cid = vec2(0.0);

  for (float dx = -1.0; dx <= 1.0; ++dx) {
    for (float dy = -1.0; dy <= 1.0; ++dy) {
      vec2 offset = vec2(dx, dy);

      vec2 n = hash22(vec2(id + offset));
      vec2 p = offset + sin(n * 10.0) * 0.5;

      float d = length(gv - p);

      if (d < minDist) {
        minDist = d;
        cid = id + offset;
      }
    }
  }

  return minDist;
}

//======================================================
// PBR
//======================================================
bool TestLightInRange(const in float lightDistance,
                      const in float cutoffDistance) {
  return any(bvec2(cutoffDistance == 0.0, lightDistance < cutoffDistance));
}
float PunctualLightIntensityToIrradianceFactor(const in float lightDistance,
                                               const in float cutoffDistance,
                                               const in float decayExponent) {
  if (decayExponent > 0.0) {
    return pow(saturate(-lightDistance / cutoffDistance + 1.0), decayExponent);
  }

  return 1.0;
}
void GetPointDirectLightIrradiance(const in PointLight pointLight,
                                   const in vec3 geometryPosition,
                                   out IncidentLight directLight) {
  vec3 L = pointLight.position - geometryPosition;
  float lightDistance = length(L);

  directLight.direction = L / lightDistance;
  if (TestLightInRange(lightDistance, pointLight.visibleDistance)) {
    directLight.color = pointLight.color;
    directLight.color *= PunctualLightIntensityToIrradianceFactor(
        lightDistance, pointLight.visibleDistance, pointLight.decay);
    directLight.visible = true;
  } else {
    directLight.color = vec3(0.0);
    directLight.visible = false;
  }
}

vec3 DiffuseColor(const in vec3 albedo, const in float metallic) {
  return mix(albedo, vec3(0.0), metallic);
}
vec3 SpecularColor(const in vec3 albedo, const in float metallic) {
  return mix(vec3(0.04), albedo, metallic);
}
vec3 DiffuseBRDF(const in vec3 diffuseColor) { return diffuseColor / PI; }

vec3 F_Schlick(const in vec3 specularColor, const in vec3 V, const in vec3 H) {
  return specularColor +
         (1.0 - specularColor) * pow(1.0 - saturate(dot(V, H)), 5.0);
}
float D_GGX(const in float a, const in float dotNH) {
  float a2 = a * a;
  float dotNH2 = dotNH * dotNH;
  float d = dotNH2 * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}
float G_SmithSchlickGGX(const in float a, const in float dotNV,
                        const in float dotNL) {
  float k = a * a * 0.5 + Epsilon;
  float gl = dotNL / (dotNL * (1.0 - k) + k);
  float gv = dotNV / (dotNV * (1.0 - k) + k);
  return gl * gv;
}
vec3 SpecularBRDF(const in IncidentLight directLight,
                  const in GeometricContext geometry,
                  const in vec3 specularColor, const in float roughnessFactor) {
  vec3 N = geometry.normal;
  vec3 V = geometry.viewDir;
  vec3 L = directLight.direction;
  vec3 H = normalize(L + V);

  float dotNL = saturate(dot(N, L));
  float dotNV = saturate(dot(N, V));
  float dotNH = saturate(dot(N, H));
  float dotVH = saturate(dot(V, H));
  float dotLV = saturate(dot(L, V));

  float a = roughnessFactor * roughnessFactor;

  vec3 F = F_Schlick(specularColor, V, H);
  float D = D_GGX(a, dotNH);
  float G = G_SmithSchlickGGX(a, dotNV, dotNL);

  return (F * G * D) / (4.0 * dotNL * dotNV + Epsilon);
}
void RE_Direct(const in IncidentLight directLight,
               const in GeometricContext geometry, const in Material material,
               out ReflectedLight reflectedLight) {
  float dotNL = saturate(dot(geometry.normal, directLight.direction));
  vec3 irradiance = dotNL * directLight.color * PI;

  vec3 diffuse = DiffuseColor(material.albedo, material.metallic);
  vec3 specular = SpecularColor(material.albedo, material.metallic);

  reflectedLight.directDiffuse += irradiance * DiffuseBRDF(diffuse);

  float roughness = map(material.roughness, 0.0, 1.0, 0.025, 1.0);
  reflectedLight.directSpecular +=
      irradiance * SpecularBRDF(directLight, geometry, specular, roughness);
}

//======================================================
// Gamma correction
//======================================================
float LinearToRec709(in float linear) {
  if (linear < 0.018) {
    return linear * 4.5;
  }

  return pow(1.099 * linear, 0.45) - 0.099;
}
vec3 LinearToRec709(in vec3 linear) {
  return vec3(LinearToRec709(linear.r), LinearToRec709(linear.g),
              LinearToRec709(linear.b));
}
float Rec709ToLinear(float rec709) {
  float Denom1 = 1.0 / 4.5;
  float Denom2 = 1.0 / 1.099;
  float Num2 = 0.099 * Denom2;
  float Exponent2 = 1.0 / 0.45;

  if (rec709 < 0.081) {
    return rec709 * Denom1;
  }

  return pow(rec709 * Denom2 + Num2, Exponent2);
}
vec3 Rec709ToLinear(in vec3 rec709) {
  return vec3(Rec709ToLinear(rec709.r), Rec709ToLinear(rec709.g),
              Rec709ToLinear(rec709.b));
}

vec3 GammaToLinear(in vec3 gamma) { return Rec709ToLinear(gamma); }
vec3 LinearToGamma(in vec3 linear) { return LinearToRec709(linear); }

//======================================================
// SDF
//======================================================
float Sign(float x) { return (x < 0.0) ? -1.0 : 1.0; }
vec2 Sign(vec2 v) {
  return vec2((v.x < 0.0) ? -1.0 : 1.0, (v.y < 0.0) ? -1.0 : 1.0);
}
float VMax(in vec2 v) { return max(v.x, v.y); }
float VMax(in vec3 v) { return max(max(v.x, v.y), v.z); }
float VMax(in vec4 v) { return max(max(max(v.x, v.y), v.z), v.w); }
float VMin(in vec2 v) { return min(v.x, v.y); }
float VMin(in vec3 v) { return min(min(v.x, v.y), v.z); }
float VMin(in vec4 v) { return min(min(min(v.x, v.y), v.z), v.w); }
void PRot(inout vec2 p, float a) { p = cos(a) * p + sin(a) * vec2(p.y, -p.x); }
vec2 PMod2(inout vec2 p, vec2 size) {
  vec2 c = floor((p + size * 0.5) / size);
  p = mod(p + size * 0.5, size) - size * 0.5;
  return c;
}
float FBoxCheap(in vec3 position, in vec3 size) {
  return VMax(abs(position) - size);
}
float FSphere(in vec3 position, in float radius) {
  return length(position) - radius;
}
float FCylinder(in vec3 position, in float radius, in float height) {
  return max(length(position.xz) - radius, abs(position.y) - height);
}
float FSlice(in vec3 position, in float rads, in float offset) {
  float first = dot(position, vec3(cos(offset), 0.0, sin(offset)));
  float second =
      dot(position, -vec3(cos(rads + offset), 0.0, sin(rads + offset)));
  return max(first, second);
}
float OpUnion(in float a, in float b) { return min(a, b); }
float OpIntersection(in float a, in float b) { return max(a, b); }
float OpDifference(in float a, in float b) { return max(a, -b); }
Hit OpUnionHit(in Hit a, in Hit b) {
  if (a.depth < b.depth)
    return a;
  return b;
}

float almostIdentity(float x, float m, float n) {
  if (x > m)
    return x;
  float a = 2.0 * n - m;
  float b = 2.0 * m - 3.0 * n;
  float t = x / m;
  return (a * t + b) * t * t + n;
}

Hit Ground(in vec3 p) { return Hit(p.z + 50.0, IdGround, p); }
Hit Heart(in vec3 p) {
  vec3 q = p * 15.0 - vec3(0.0, 0.0, 0.0);
  vec2 c = PMod2(q.xy, vec2(125.0));
  PRot(q.xz, c.x + c.y);

  q.y -= HeartSize;

  q *= 1.0 - 0.2 * vec3(1.0, 0.5, 1.0);
  q.x = abs(q.x);
  q.z *= (2.0 - q.y / 25.0);

  q.x = almostIdentity(q.x, 1.0, 0.3);

  q.y = 4.0 + q.y * 1.2 - q.x * sqrt(max((20.0 - q.x) / 15.0, 0.0));
  float d = length(q) - HeartSize;
  d /= 2.0;
  d /= HeartSize;

  float guard = abs(-FBoxCheap(q, vec3(HeartSize * 0.5))) + HeartSize * 0.1;
  d = OpUnion(d, guard * 0.01);

  return Hit(d, IdHeart, q);
}

Hit GetSceneData(in vec3 position) {
  vec3 p = position;

  Hit ground = Ground(p);
  Hit heart = Heart(p);

  Hit result = ground;
  result = OpUnionHit(result, heart);

  return result;
}

Material GetMaterial(in vec3 position, in vec3 localPosition, in vec3 normal,
                     in int id) {
  Material material;
  material.albedo = vec3(0.0);
  material.roughness = 0.0;
  material.metallic = 0.0;

  switch (id) {
  case IdGround:
    material.albedo = mix(GroundAlbedo.rgb, GroundAlbedo2.rgb,
                          voronoi(localPosition.xz * 0.1));
    material.roughness = 0.95;
    material.metallic = 0.0;
    break;
  case IdHeart:
    material.albedo = HeartAlbedo.rgb;
    material.roughness = 0.35;
    material.metallic = 0.0;
    break;
  }

  material.albedo = max(GammaToLinear(material.albedo), 0.01);
  return material;
}

//======================================================
// Ray Marching
//======================================================
Hit RayMarch(const in Ray ray) {
  const int steps = RayMarchingSteps;

  Hit hit;
  float currentDepth = 0.0;
  vec3 currentPosition;

  for (int iterations = ZERO; iterations < steps; ++iterations) {
    currentPosition = ray.direction * currentDepth + ray.origin;
    hit = GetSceneData(currentPosition);
    if (abs(hit.depth) < (RayMarchingSurfaceDistance) ||
        currentDepth > RayMarchingMaxDistance) {
      break;
    }
    currentDepth += hit.depth;
  }

  return Hit(currentDepth, hit.id, hit.localPosition);
}

//======================================================
// Normal and AO Calculations
//======================================================
vec3 CalcNormal(in vec3 position) {
  const vec2 e = vec2(1.0, -1.0) * 2e-4;
  return normalize(e.xyy * GetSceneData(position + e.xyy).depth +
                   e.yyx * GetSceneData(position + e.yyx).depth +
                   e.yxy * GetSceneData(position + e.yxy).depth +
                   e.xxx * GetSceneData(position + e.xxx).depth);
}

//======================================================
// Camera
//======================================================
Ray GetRay(in vec2 uv) {
#ifdef TAKOYAKI_
  vec3 lookAtPosition = iCameraTarget;
  vec3 cameraPosition = iCameraOrigin;
#else
  float offset = HeartSize / 16.0;
  vec3 lookAtPosition = vec3(0.0, offset, 0.0);
  vec3 cameraPosition =
      vec3(0.0, 0.0, -1.0) * CameraDistance + vec3(0.0, offset, 0.0);
#endif

  vec3 forward = normalize(lookAtPosition - cameraPosition);
  vec3 right = normalize(vec3(forward.z, 0.0, -forward.x));
  vec3 up = normalize(cross(forward, right));

  return Ray(cameraPosition, normalize(forward + FocalLength * uv.x * right +
                                       FocalLength * uv.y * up));
}

const int NumLights = 4;
const PointLight[] PointLights = PointLight[](
    PointLight(LightPositionLeft, LightColorLeft.rgb *LightColorLeft.a,
               LightDistanceDecayLeft.x, LightDistanceDecayLeft.y),
    PointLight(LightPositionRight, LightColorRight.rgb *LightColorRight.a,
               LightDistanceDecayRight.x, LightDistanceDecayRight.y),
    PointLight(LightPositionBack, LightColorBack.rgb *LightColorBack.a,
               LightDistanceDecayBack.x, LightDistanceDecayBack.y));

vec3 GetColor(in vec2 uv) {
  Ray ray = GetRay(uv);

  Hit hit = RayMarch(ray);

#ifdef DEBUG_ITERATIONS
  return mix(vec3(0, 1, 0), vec3(1, 0, 0),
             pow(float(iterations) / float(RayMarchingSteps), 0.5));
#endif

  if (hit.depth > RayMarchingMaxDistance) {
    return vec3(0.0);
  }

#ifdef DEBUG_LOCAL_POSITION
  return abs(hit.localPosition);
#endif

  vec3 position = ray.direction * hit.depth + ray.origin;
  vec3 normal = CalcNormal(position);
  GeometricContext geometry =
      GeometricContext(position, normal, normalize(ray.origin - position));

#ifdef DEBUG_NORMALS
  return normal * 0.5 + 0.5;
#endif

  Material material = GetMaterial(position, hit.localPosition, normal, hit.id);

#ifdef DEBUG_ALBEDO
  return material.albedo;
#endif
#ifdef DEBUG_ROUGHNESS
  return vec3(material.roughness);
#endif
#ifdef DEBUG_METALLIC
  return vec3(material.metallic);
#endif

  ReflectedLight reflectedLight =
      ReflectedLight(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
  IncidentLight directLight;

  for (int i = ZERO; i < NumLights; ++i) {
    GetPointDirectLightIrradiance(PointLights[i], geometry.position,
                                  directLight);
    if (directLight.visible) {
      vec3 L = PointLights[i].position - geometry.position;
      RE_Direct(directLight, geometry, material, reflectedLight);
    }
  }

  vec3 diffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
  vec3 specular =
      reflectedLight.directSpecular + reflectedLight.indirectSpecular;

#ifdef DEBUG_DIFFUSE
  return diffuse;
#endif
#ifdef DEBUG_SPECULAR
  return specular;
#endif

  vec3 color = vec3(0.0);
  color += diffuse;
  color += specular;
  color += AmbientLight.rgb * AmbientLight.a * material.albedo;

  return color;
}

#define AA
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec2 uv = (2.0 * (fragCoord)-iResolution.xy) / iResolution.y;
  vec3 color = vec3(0.0);

#ifdef AA
  float spread = 1.0;
  color += GetColor(
      (2.0 * (fragCoord + (vec2(0, 0) / 2.0 - 0.5) * spread) - iResolution.xy) /
      iResolution.y);
  color += GetColor(
      (2.0 * (fragCoord + (vec2(1, 0) / 2.0 - 0.5) * spread) - iResolution.xy) /
      iResolution.y);
  color += GetColor(
      (2.0 * (fragCoord + (vec2(0, 1) / 2.0 - 0.5) * spread) - iResolution.xy) /
      iResolution.y);
  color += GetColor(
      (2.0 * (fragCoord + (vec2(1, 1) / 2.0 - 0.5) * spread) - iResolution.xy) /
      iResolution.y);
  color /= 4.0;
#else
  color += GetColor(uv);
#endif

  float ditherValue =
      DitherPattern[int(fragCoord.y) % 4][int(fragCoord.x) % 4] / 16.0;

  color += (1.0 / 255.0) * ditherValue;

  color = LinearToGamma(color);

  // vignette
  {
    vec2 uv = fragCoord.xy / iResolution.xy;
    uv *= 1.0 - uv.yx;
    float vignette = uv.x * uv.y * 20.0;
    vignette = pow(vignette, 0.15);
    color *= saturate(vignette);
  }

  fragColor = vec4(color, 1.0);
}