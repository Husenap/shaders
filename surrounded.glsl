// Available uniforms
//#=============================#
//* uniform float iTime;        *
//* uniform vec2 iResolution;   *
//* uniform int iFrame;         *
//* uniform vec3 iCameraOrigin; *
//* uniform vec3 iCameraTarget; *
//#=============================#

#ifndef TAKOYAKI
const float FocalLength = 0.170;
const float CameraHeight = 24.813;
const float CameraDistance = 17.236;
const float WallThickness = 0.1;
const float WallCutout = 1.5;
const float BallSize = 0.75;
const vec4 GroundAlbedo = vec4(0.292, 0.892, 0.768, 1.000);
const vec4 GroundAlbedo2 = vec4(0.627, 0.219, 1.000, 1.000);
const vec4 WallAlbedo = vec4(1.0, 1.0, 1.0, 1.0);
const vec4 BallAlbedo = vec4(0.962, 0.962, 0.962, 1.0);

const vec3 LightPositionLeft = vec3(-92.3, 66.2, -24.2);
const vec3 LightPositionRight = vec3(70.5, 68.4, -14.0);
const vec3 LightPositionBack = vec3(0.0, 152.5, 496.7);

const vec4 LightColorLeft = vec4(0.627, 0.219, 1.000, 0.000);
const vec4 LightColorRight = vec4(0.330, 0.208, 1.000, 1.306);
const vec4 LightColorBack = vec4(0.622, 0.523, 1.000, 0.341);

const vec2 LightDistanceDecayLeft = vec2(427.2, 1.0);
const vec2 LightDistanceDecayRight = vec2(392.7, 1.0);
const vec2 LightDistanceDecayBack = vec2(2934.0, 1.0);

const vec4 AmbientLight = vec4(0.233, 0.090, 0.900, 0.663);

const float ShadowBias = 0.4;
#endif

//======================================================
// CONSTANTS
//======================================================
const float Epsilon = 1e-4;

const int RayMarchingSteps = 256;
const float RayMarchingSurfaceDistance = 5e-4;
const float RayMarchingMaxDistance = 1000.0;
const float RayMarchingMaxBounces = 1.0;

const float AOMaxSteps = 8.0;
const float AOSampleLength = 1.0 / 7.0;

const int ShadowSteps = 32;

const mat4 DitherPattern = mat4(0.0, 12., 3.0, 15., 8.0, 4.0, 11., 7.0, 2.0,
                                14., 1.0, 13., 10., 6.0, 9.0, 5.0);

const int IdGround = 0;
const int IdWall = 1;
const int IdBall = 2;

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
  const vec3 L = pointLight.position - geometryPosition;
  const float lightDistance = length(L);

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
  const float a2 = a * a;
  const float dotNH2 = dotNH * dotNH;
  const float d = dotNH2 * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}
float G_SmithSchlickGGX(const in float a, const in float dotNV,
                        const in float dotNL) {
  const float k = a * a * 0.5 + Epsilon;
  const float gl = dotNL / (dotNL * (1.0 - k) + k);
  const float gv = dotNV / (dotNV * (1.0 - k) + k);
  return gl * gv;
}
vec3 SpecularBRDF(const in IncidentLight directLight,
                  const in GeometricContext geometry,
                  const in vec3 specularColor, const in float roughnessFactor) {
  const vec3 N = geometry.normal;
  const vec3 V = geometry.viewDir;
  const vec3 L = directLight.direction;
  const vec3 H = normalize(L + V);

  const float dotNL = saturate(dot(N, L));
  const float dotNV = saturate(dot(N, V));
  const float dotNH = saturate(dot(N, H));
  const float dotVH = saturate(dot(V, H));
  const float dotLV = saturate(dot(L, V));

  const float a = roughnessFactor * roughnessFactor;

  const vec3 F = F_Schlick(specularColor, V, H);
  const float D = D_GGX(a, dotNH);
  const float G = G_SmithSchlickGGX(a, dotNV, dotNL);

  return (F * G * D) / (4.0 * dotNL * dotNV + Epsilon);
}
void RE_Direct(const in IncidentLight directLight,
               const in GeometricContext geometry, const in Material material,
               out ReflectedLight reflectedLight) {
  const float dotNL = saturate(dot(geometry.normal, directLight.direction));
  const vec3 irradiance = dotNL * directLight.color * PI;

  const vec3 diffuse = DiffuseColor(material.albedo, material.metallic);
  const vec3 specular = SpecularColor(material.albedo, material.metallic);

  reflectedLight.directDiffuse += irradiance * DiffuseBRDF(diffuse);

  const float roughness = map(material.roughness, 0.0, 1.0, 0.025, 1.0);
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
  const float Denom1 = 1.0 / 4.5;
  const float Denom2 = 1.0 / 1.099;
  const float Num2 = 0.099 * Denom2;
  const float Exponent2 = 1.0 / 0.45;

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

Hit Ground(in vec3 p) { return Hit(p.y, IdGround, p); }
Hit Ball(in vec3 p) {
  const vec3 q = p - vec3(0.0, BallSize, 0.0);

  float d = FSphere(q, BallSize);

  for (float radius = 4.0; radius < 12.0; radius += 2.0) {
    vec3 w = q;
    PRot(w.xz, radius * 0.7 + TIME * sin(radius) / radius);
    w -= vec3(0.0, 0.0, radius);
    d = OpUnion(d, FSphere(w, BallSize));
  }

  return Hit(d, IdBall, q);
}
Hit Wall(in vec3 p) {
  const vec3 q = p - vec3(0.0, BallSize, 0.0);

  float d = RayMarchingMaxDistance;
  for (float radius = 3.0; radius < 12.0; radius += 2.0) {
    const float inner = FCylinder(q, radius - WallThickness, BallSize * 2.0);
    const float outer = FCylinder(q, radius, BallSize);

    float cutoutBox = RayMarchingMaxDistance;
    for (float a = 0.0; a < 3.0; ++a) {
      float cutout = WallCutout;
      cutoutBox =
          OpUnion(cutoutBox, FSlice(q, cutout / radius,
                                    a * 2.0 * PI / 3.0 - cutout / radius * 0.5 +
                                        radius * 0.7));
    }

    d = OpUnion(d, OpDifference(OpDifference(outer, inner), cutoutBox));
  }

  return Hit(d, IdWall, q);
}

Hit GetSceneData(in vec3 position) {
  vec3 p = position;

  Hit ground = Ground(p);
  Hit ball = Ball(p);
  Hit wall = Wall(p);

  Hit result = ground;
  result = OpUnionHit(result, ball);
  result = OpUnionHit(result, wall);

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
  case IdWall:
    material.albedo = WallAlbedo.rgb;
    material.roughness = 0.95;
    material.metallic = 0.0;
    break;
  case IdBall:
    material.albedo = BallAlbedo.rgb;
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

float CalcAO(in vec3 position, in vec3 normal) {
  float k = 1.0;
  float occlusion = 0.0;
  for (float i = 1.0; i <= AOMaxSteps; ++i) {
    float sampleLength = AOSampleLength * i;
    float sampleDistance = GetSceneData(normal * sampleLength + position).depth;
    occlusion += k * (sampleLength - sampleDistance);
    k *= 0.5;
  }
  return max(1.0 - occlusion, 0.0);
}

float CalcShadow(in vec3 ro, in vec3 rd, in float maxT) {
  float res = 1.0;
  float t = 0.01;
  float ph = 1e10;

  for (int i = ZERO; i < ShadowSteps; ++i) {
    float h = GetSceneData(rd * t + ro).depth;

    if (h < RayMarchingSurfaceDistance) {
      return 0.0;
    }

    float y = h * h / (2. * ph);
    float d = sqrt(h * h - y * y);
    res = min(res, 5. * d / max(0., t - y));
    ph = h;

    t += h * 0.95;

    if (res < RayMarchingSurfaceDistance || t >= maxT)
      break;
  }

  return saturate(res);
}

//======================================================
// Camera
//======================================================
Ray GetRay(in vec2 uv) {
#ifdef TAKOYAKI_
  vec3 lookAtPosition = iCameraTarget;
  vec3 cameraPosition = iCameraOrigin;
#else
  vec3 lookAtPosition = vec3(0.0, BallSize, 0.0);
  vec3 cameraPosition =
      vec3(0.0, 0.0, -1.0) * CameraDistance + vec3(0.0, CameraHeight, 0.0);
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
      float shadow = CalcShadow(position + normal * ShadowBias *
                                               hash33(geometry.position).x,

                                normalize(L), length(L));

      directLight.color *= shadow;
      RE_Direct(directLight, geometry, material, reflectedLight);
    }
  }

  const vec3 diffuse =
      reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
  const vec3 specular =
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

  float ambientOcclusion = CalcAO(position, normal);
#ifdef DEBUG_AO
  return vec3(ambientOcclusion);
#endif
  color *= ambientOcclusion;

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

  fragColor = vec4(color, 1.0);
}