// Available uniforms
//#=============================#
//* uniform float iTime;        *
//* uniform vec2 iResolution;   *
//* uniform int iFrame;         *
//* uniform vec3 iCameraOrigin; *
//* uniform vec3 iCameraTarget; *
//#=============================#

//======================================================
// TAKOYAKI UNIFORMS
//======================================================
#ifndef TAKOYAKI
const float FocalLength = 0.5;

#endif

//======================================================
// CONSTANTS
//======================================================
const float Epsilon = 1e-4;

const int RayMarchingSteps = 128;
const float RayMarchingSurfaceDistance = 5e-4;
const float RayMarchingMaxDistance = 50.0;
const float RayMarchingMaxBounces = 1.0;

const float AOMaxSteps = 4.0;
const float AOSampleLength = 1.0 / 7.0;

const int IdMushroom = 0;
const int IdGround = 1;

const mat4 DitherPattern = mat4(0.0, 12., 3.0, 15., 8.0, 4.0, 11., 7.0, 2.0,
                                14., 1.0, 13., 10., 6.0, 9.0, 5.0);

const float PI = 3.1415926535;
const float DegToRad = PI / 180.0;

#define TIME (iTime * 15.0)

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
//#define DEBUG_BOUNCES

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
float map(const in float value, const in float low1, const in float high1,
          const in float low2, const in float high2) {
  return low2 + (value - low1) * (high2 - low2) / (high1 - low1);
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
// Camera
//======================================================
Ray GetRay(in vec2 uv) {
#ifdef TAKOYAKI_
  vec3 lookAtPosition = iCameraTarget;
  vec3 cameraPosition = iCameraOrigin;
#else
  vec3 lookAtPosition = vec3(0.0, 0.0, 0.0);
  vec3 cameraPosition = vec3(cos(TIME), 0.1, sin(TIME)) * 5.5;
#endif

  vec3 forward = normalize(lookAtPosition - cameraPosition);
  vec3 right = normalize(vec3(forward.z, 0.0, -forward.x));
  vec3 up = normalize(cross(forward, right));

  return Ray(cameraPosition, normalize(forward + FocalLength * uv.x * right +
                                       FocalLength * uv.y * up));
}

const int NumLights = 2;
const PointLight[] PointLights = PointLight[](
    PointLight(LightPosition, LightColor.rgb, LightDistanceDecay.x,
               LightDistanceDecay.y),
    PointLight(vec3(-LightPosition.x, LightPosition.yz), LightColor.rgb,
               LightDistanceDecay.x, LightDistanceDecay.y));

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

float FSphere(in vec3 position, in float radius) {
  return length(position) - radius;
}
float FBoxCheap(in vec3 position, in vec3 size) {
  return VMax(abs(position) - size);
}
float FBoxCheap2(in vec2 position, in vec2 size) {
  return VMax(abs(position) - size);
}
float FBox(in vec3 position, in vec3 size) {
  vec3 q = abs(position) - size;
  return length(max(q, 0.0)) + VMax(min(q, 0.0));
}
float FBoxRound(in vec3 position, in vec3 size, in float radius) {
  vec3 q = abs(position) - size + radius;
  return length(max(q, 0.0)) + VMax(min(q, 0.0)) - radius;
}
float FCylinder(in vec3 position, in float radius, in float height) {
  return max(length(position.xz) - radius, abs(position.y) - height);
}
float FTorus(in vec3 p, vec2 t) {
  vec2 q = vec2(length(p.xz) - t.x, p.y);
  return length(q) - t.y;
}
float FCapsule(vec3 p, vec3 a, vec3 b, float r) {
  vec3 pa = p - a;
  vec3 ba = b - a;
  float h = saturate(dot(pa, ba) / dot(ba, ba));
  return length(pa - ba * h) - r;
}

float OpUnion(in float a, in float b) { return min(a, b); }
float OpIntersection(in float a, in float b) { return max(a, b); }
float OpDifference(in float a, in float b) { return max(a, -b); }
float OpSmoothUnion(in float a, in float b, in float k) {
  float h = max(k - abs(a - b), 0.0) / k;
  return min(a, b) - h * h * k * (1.0 / 4.0);
}

void PRot(inout vec2 p, float a) { p = cos(a) * p + sin(a) * vec2(p.y, -p.x); }
float PMod1(inout float position, in float size) {
  float halfSize = size * 0.5;
  float c = floor((position + halfSize) / size);
  position = mod(position + halfSize, size) - halfSize;
  return c;
}
float PModInterval1(inout float position, in float size, in float start,
                    in float stop) {
  float halfSize = size * 0.5;
  float c = floor((position + halfSize) / size);
  position = mod(position + halfSize, size) - halfSize;
  if (c > stop) {
    position += size * (c - stop);
    c = stop;
  }
  if (c < start) {
    position += size * (c - start);
    c = start;
  }
  return c;
}
float PMirror(inout float position, float dist) {
  float s = Sign(position);
  position = abs(position) - dist;
  return s;
}

Hit OpUnionHit(in Hit a, in Hit b) {
  if (a.depth < b.depth)
    return a;
  return b;
}

Hit Mushroom(vec3 p) {
  float d = FSphere(p, 1.0);

  if (d < 1.0) {
    float t = 1.0 - saturate(p.y);
    float y = t * t * (3 - 2 * t) + sqrt(t) * (1.0 - t);

    vec3 sway = vec3(cos(TIME), 0.0, sin(TIME)) * 0.1;

    vec3 q1 = p - sway;
    d = length(q1.xz) - y;
    d = max(d, -p.y);
    d = max(d, p.y - 1);

    float capAngle = atan(p.z, p.x) * 20.0;
    d -= cos(capAngle) * 0.01 * saturate(exp(-p.y * 10.0));
    d = OpSmoothUnion(d, FTorus(q1 - vec3(0.0, 0.1, 0.0), vec2(1.0, 0.15)),
                      0.05);

    vec3 q = p - vec3(cos(p.y * 4.3 + TIME), 0.0, sin(p.y * 4.3 + TIME)) * 0.1 *
                     (p.y + 1.0);

    float a = atan(q.z, q.x) * 20.0;

    float stem = FCapsule(q, vec3(0.0, -1.0, 0.0), vec3(0.0, 0.0, 0.0),
                          0.3 + 0.001 * cos(a));

    d = min(d, stem);
  }

  return Hit(d, IdMushroom, p);
}

Hit Ground(vec3 p) { return Hit(p.y + 1, IdGround, p); }

Hit GetSceneData(in vec3 position) {
  vec3 p = position;

  Hit mushroom = Mushroom(p);
  Hit ground = Ground(p);

  Hit result = mushroom;
  result = OpUnionHit(result, ground);

  return result;
}

//======================================================
// Material
//======================================================
Material GetMaterial(in vec3 position, in vec3 localPosition, in vec3 normal,
                     in int id) {
  Material material;
  material.albedo = vec3(0.0);
  material.roughness = 0.0;
  material.metallic = 0.0;

  switch (id) {
  case IdMushroom:
    material.albedo = vec3(0.9, 0.9, 0.9);
    material.roughness = 0.95;
    material.metallic = 0.0;
    break;
  }

  material.albedo = max(GammaToLinear(material.albedo), 0.01);
  return material;
}

//======================================================
// Ray Marching
//======================================================
Hit RayMarch(const in Ray ray, const in float bounce, inout int iterations) {
  const int steps = int(float(RayMarchingSteps) / (1.0 + bounce));

  Hit hit;
  float currentDepth = 0.0;
  vec3 currentPosition;

  for (iterations = 0; iterations < steps; ++iterations) {
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

//======================================================
// Lighting Calculations
//======================================================
vec3 GetColor(in vec2 uv) {
  Ray ray = GetRay(uv);

  vec3 result = vec3(0.0);
  vec3 carry = vec3(1.0);
  float lastRoughness = 0.0;

  float bounce = 0;
  for (; bounce <= RayMarchingMaxBounces; ++bounce) {
    int iterations = 0;
    const Hit hit =
        RayMarch(ray, bounce / (1.0 - 0.5 * lastRoughness), iterations);

#ifdef DEBUG_ITERATIONS
    return mix(vec3(0, 1, 0), vec3(1, 0, 0),
               pow(float(iterations) / float(RayMarchingSteps), 0.5));
#endif

    if (hit.depth > RayMarchingMaxDistance) {
      result += carry * (vec3(0.25) + 0.3 * ray.direction.y);
      break;
    }

#ifdef DEBUG_LOCAL_POSITION
    return abs(hit.localPosition);
#endif

    const vec3 position = ray.direction * hit.depth + ray.origin;
    const vec3 normal = CalcNormal(position);
    const GeometricContext geometry =
        GeometricContext(position, normal, normalize(ray.origin - position));

#ifdef DEBUG_NORMALS
    return abs(normal);
#endif

    const Material material =
        GetMaterial(position, hit.localPosition, normal, hit.id);

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

    for (int i = 0; i < NumLights; ++i) {
      GetPointDirectLightIrradiance(PointLights[i], geometry.position,
                                    directLight);
      if (directLight.visible) {
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
    color += 0.01 * material.albedo;

    vec3 albedo = vec3(0.18);

    const vec3 n = normal;
    float sun_dif = saturate(dot(n, normalize(vec3(0.8, 0.4, 0.2))));
    float sky_dif = saturate(0.5 + 0.5 * dot(n, vec3(0.0, 1.0, 0.0)));
    float bou_dif = saturate(0.5 + 0.5 * dot(n, vec3(0.0, -1.0, 0.0)));

    color = vec3(0.0);
    color += albedo * vec3(7.0, 4.5, 3.0) * sun_dif;
    color += albedo * vec3(0.5, 0.8, 0.9) * sky_dif;
    color += albedo * vec3(0.7, 0.3, 0.2) * bou_dif;

    if (bounce < 1.0) {
      const float ambientOcclusion = CalcAO(position, normal);
#ifdef DEBUG_AO
      return vec3(ambientOcclusion);
#endif
      color *= ambientOcclusion;
    }

    result += color * carry;

    lastRoughness = material.roughness;
    carry = material.albedo * carry * (1.0 - material.roughness);
    if (VMax(carry) < 0.05) {
      break;
    }

    ray.direction -= 2.0 * dot(ray.direction, normal) * normal;
    ray.origin = position + ray.direction * RayMarchingSurfaceDistance * 2.0 *
                                (bounce + 1.0);
  }

#ifdef DEBUG_BOUNCES
  return mix(vec3(0, 1, 0), vec3(1, 0, 0),
             pow(bounce / RayMarchingMaxBounces, 0.5));
#endif

  return result;
}

//======================================================
// Main
//======================================================
#define ZERO 0
#define AA 1
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec3 linearColor = vec3(0.0);

#if AA > 1
  for (int m = ZERO; m < AA; m++) {
    for (int n = ZERO; n < AA; n++) {
      vec2 o = vec2(float(m), float(n)) / float(AA) - 0.5;
      vec2 uv = (2.0 * (fragCoord + o) - iResolution.xy) / iResolution.y;
#else
  vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
#endif
      linearColor += GetColor(uv);
#if AA > 1
    }
  }
  linearColor /= float(AA * AA);
#endif

  vec3 color = LinearToGamma(linearColor);

  fragColor = vec4(color, 1.0);
}