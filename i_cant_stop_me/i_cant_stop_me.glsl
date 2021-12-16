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

const float BeamWidth = 4.5;
const float BeamThickness = 0.15;
const float BeamRadius = 1.2;
const vec4 BeamColor = vec4(0.623, 0.942, 0.758, 1.000);

const vec4 Pantone712C = vec4(0.988, 0.784, 0.608, 1.000);
const vec4 Pantone812C = vec4(1.000, 0.373, 0.635, 1.000);
const vec4 Pantone636C = vec4(0.545, 0.827, 0.902, 1.000);

const vec3 PillarSize = vec3(0.8, 2.5, 0.4);
const vec4 PillarBody = vec4(1.000, 0.805, 0.365, 1.000);
const vec4 PillarRed = vec4(1.000, 0.327, 0.327, 1.000);
const vec4 PillarGreen = vec4(0.427, 1.000, 0.616, 1.000);
const vec4 PillarBottom = vec4(0.815, 0.661, 0.379, 1.000);
const vec4 PillarPoster = vec4(0.135, 0.137, 0.169, 1.000);
const vec4 PillarEdge = vec4(0.869, 0.869, 0.869, 1.000);
const float PillarSpacing = 4.0;
const float PillarCount = 4.0;

const vec3 StairsStepSize = vec3(4.0, 0.2, 0.25);

const float MainWidth = 4.5;
const float SafetyLineSize = 0.3;

const float PlatformWidth = 2.5;
const vec4 PlatformRoofColor = vec4(1.000, 0.895, 0.658, 1.000);

const float RoundRoofDepth = 0.5;
const vec4 RoundRoofColor = vec4(0.165, 0.209, 0.612, 1.000);

const vec2 SkyWindowSize = vec2(1.0, 5.0);
const vec4 SkyWindowColor = vec4(1.000, 0.315, 0.963, 1.000);
const vec4 SkyWindowWhiteColor = vec4(0.885, 0.885, 0.885, 1.000);
const vec4 SkyWindowBlackColor = vec4(0.177, 0.177, 0.177, 1.000);
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

const int IdBeam = 0;
const int IdPillarBody = 1;
const int IdPillarRed = 2;
const int IdPillarGreen = 3;
const int IdPillarBottom = 4;
const int IdPillarPoster = 5;
const int IdPillarEdge = 6;
const int IdBench = 7;
const int IdStairs = 8;
const int IdMainFloor = 9;
const int IdPlatformFloor = 10;
const int IdSafetyLine = 11;
const int IdPlatformRoof = 12;
const int IdRoundRoof = 13;
const int IdSkyWindow = 14;
const int IdSkyWindowRingWhite = 15;
const int IdSkyWindowRingBlack = 16;
const int IdTrainTunnel = 17;
const int IdTrainTunnelWall = 18;
const int IdTrain = 19;
const int IdTrainSupportBeam = 20;

float PlatformLength() {
  return PillarSpacing * (PillarCount - 1.0) + PillarSize.z * 0.5;
}

const mat4 DitherPattern = mat4(0.0, 12., 3.0, 15., 8.0, 4.0, 11., 7.0, 2.0,
                                14., 1.0, 13., 10., 6.0, 9.0, 5.0);

const float PI = 3.1415926535;
const float DegToRad = PI / 180.0;

#define TIME (iTime)

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
  vec3 lookAtPosition = vec3(0.0, 1.0, 10.0);
  vec3 cameraPosition = vec3(cos(TIME), 0.5, sin(TIME)) * 2.5;
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

float OpUnion(in float a, in float b) { return min(a, b); }
float OpIntersection(in float a, in float b) { return max(a, b); }
float OpDifference(in float a, in float b) { return max(a, -b); }

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

Hit Beam(in vec3 position) {
  const float Side = MainWidth * 0.5 + SafetyLineSize;
  const float Radius = BeamRadius;
  const float Thickness = BeamThickness;
  const float Depth = 0.05;

  vec3 q = position;
  q.y += Side;
  q.y -= PillarSize.y + BeamRadius + RoundRoofDepth;
  PModInterval1(q.z, PillarSpacing, 1.0, PillarCount - 1.0);

  float mainBeam = FBoxRound(q, vec3(Side, Side, Side * 2.0), Radius);
  float cutoutBeam =
      FBoxRound(q - vec3(0.0, -Thickness, 0.0),
                vec3(Side - Thickness, Side, Side * 2.0), Radius - Thickness);
  float cutoutBox =
      FBoxCheap(q - vec3(0.0, Side, 0.0), vec3(Side, Radius, Depth));

  vec3 cylinderOffset = vec3(0.0, -Thickness * 0.5 + Side, 0.0);
  vec3 cylinderPosition = q - cylinderOffset;
  PMod1(cylinderPosition.x, Side - Radius);

  float cutoutCylinder = FCylinder(cylinderPosition.xzy, Thickness * 0.2, 1.0);

  vec3 cylinderPosition1 = q - cylinderOffset;
  PMirror(cylinderPosition1.x, 0.0);
  cylinderPosition1.x -= Side - Thickness * 0.5;
  cylinderPosition1.xy += (Radius - Thickness * 0.5);

  cutoutCylinder =
      OpUnion(FCylinder((cylinderPosition1 -
                         vec3(cos(20.0 * DegToRad), sin(20.0 * DegToRad), 0.0) *
                             (Radius - Thickness * 0.5))
                            .xzy,
                        Thickness * 0.2, 1.0),
              cutoutCylinder);
  cutoutCylinder =
      OpUnion(FCylinder((cylinderPosition1 -
                         vec3(cos(70.0 * DegToRad), sin(70.0 * DegToRad), 0.0) *
                             (Radius - Thickness * 0.5))
                            .xzy,
                        Thickness * 0.2, 1.0),
              cutoutCylinder);

  float d = OpDifference(mainBeam, cutoutBeam);
  d = OpIntersection(d, cutoutBox);
  d = OpDifference(d, cutoutCylinder);

  return Hit(d, IdBeam, q);
}

Hit Pillar(in vec3 position) {
  vec3 pillarSize = PillarSize * 0.5;
  vec3 lipSize = vec3(pillarSize.x + 0.025, 0.05, pillarSize.z + 0.025);

  position.y -= pillarSize.y;

  Hit body = Hit(FBoxCheap(position, pillarSize), IdPillarBody, position);
  Hit green = Hit(
      FBoxRound(position - vec3(0.0, pillarSize.y - lipSize.y * 0.9, 0.0),
                lipSize, 0.01),
      IdPillarGreen, position - vec3(0.0, pillarSize.y - lipSize.y * 0.9, 0.0));
  Hit red = Hit(
      FBoxRound(position + vec3(0.0, pillarSize.y - lipSize.y * 3.0, 0.0),
                lipSize, 0.01),
      IdPillarRed, position + vec3(0.0, pillarSize.y - lipSize.y * 3.0, 0.0));
  Hit bottom =
      Hit(FBoxRound(position + vec3(0.0, pillarSize.y - lipSize.y * 1.25, 0.0),
                    lipSize + vec3(0.025), 0.01),
          IdPillarBottom,
          position + vec3(0.0, pillarSize.y - lipSize.y * 1.25, 0.0));

  vec3 posterPosition = position;
  PMirror(posterPosition.x, pillarSize.x);
  posterPosition.y -= pillarSize.y / 3.0;
  Hit poster = Hit(FBoxCheap(posterPosition, vec3(0.01, pillarSize.y / 6.0,
                                                  pillarSize.z * 0.5)),
                   IdPillarPoster, posterPosition);

  vec3 edgePosition = position;
  PMirror(edgePosition.x, pillarSize.x);
  PMirror(edgePosition.z, pillarSize.z);
  Hit edge =
      Hit(FBoxRound(edgePosition, vec3(0.01, pillarSize.y, 0.01), 0.0025),
          IdPillarEdge, edgePosition);

  Hit result = body;
  result = OpUnionHit(green, result);
  result = OpUnionHit(red, result);
  result = OpUnionHit(bottom, result);
  result = OpUnionHit(poster, result);
  result = OpUnionHit(edge, result);

  return result;
}
Hit Pillars(in vec3 position) {
  PModInterval1(position.z, PillarSpacing, 0.0, PillarCount - 1.0);
  PMirror(position.x, (MainWidth + PlatformWidth) * 0.5 + SafetyLineSize);

  Hit result = Pillar(position);

  return result;
}

Hit Bench(in vec3 position) {
  vec3 benchSize = vec3(0.5, 0.5, PillarSpacing * 0.5) * 0.5;
  float benchThickness = 0.0325;

  position.z += PillarSpacing * 0.5;
  PModInterval1(position.z, PillarSpacing, 1.0, 2.0);
  position.y -= benchSize.y;
  PMirror(position.x, (MainWidth + PlatformWidth) * 0.5 + SafetyLineSize);

  vec3 grillPosition = position;
  PMod1(grillPosition.x, benchThickness * 1.1);

  float bench = FBoxCheap(position, benchSize);
  float benchCutout =
      FBoxCheap(position - vec3(0.0, -1.0, 0.0) * benchThickness,
                benchSize - vec3(-1.0, 0.0, 1.0) * benchThickness);
  float grillCutout =
      FBoxCheap(grillPosition, vec3(benchThickness * 0.3, 10.0, 10.0));

  float d = OpDifference(bench, benchCutout);
  d = OpDifference(d, grillCutout);

  return Hit(d, IdBench, position);
}

Hit StairsRail(in vec3 position) {
  position.z -= PlatformLength();
  PMirror(position.x, 0.0);
  position.x -= 0.2;
  PModInterval1(position.x, StairsStepSize.x * 0.5 - 0.3, 0.0, 1.0);

  const vec3 supportSize = vec3(0.02, StairsStepSize.y * 1.9, 0.02);

  vec3 railPosition = position;
  railPosition.z += 11.5 * StairsStepSize.z;
  railPosition.y -= StairsStepSize.y + supportSize.y;

  vec3 handleStartPosition = railPosition;

  float d = FBoxCheap(railPosition, supportSize);
  for (float i = 0.0; i < 3.0; ++i) {
    railPosition.z -= 2.0 * StairsStepSize.z;
    railPosition.y -= 2.0 * StairsStepSize.y;
    d = OpUnion(FBoxCheap(railPosition, supportSize), d);
  }

  vec3 handlePosition = (handleStartPosition + railPosition) * 0.5;
  handlePosition *= mat3(1.0, 0.0, 0.0, 0.0, 1.0,
                         -StairsStepSize.y / StairsStepSize.z, 0.0, 0.0, 1.0);

  d = OpUnion(
      FBoxCheap(handlePosition - vec3(0.0, supportSize.y,
                                      StairsStepSize.z * 3.0 + supportSize.z +
                                          StairsStepSize.z * 0.56),
                vec3(supportSize.x, supportSize.x, StairsStepSize.z * 0.56)),
      d);

  PModInterval1(handlePosition.y, supportSize.y, 0.0, 1.0);

  d = OpUnion(
      FBoxCheap(handlePosition, vec3(supportSize.x, supportSize.x,
                                     StairsStepSize.z * 3.0 + supportSize.z)),
      d);

  railPosition.z -= 3.0 * StairsStepSize.z;
  railPosition.y -= 3.0 * StairsStepSize.y;
  railPosition.y -= StairsStepSize.y - supportSize.y;
  d = OpUnion(FBoxCheap(railPosition,
                        vec3(supportSize.x, StairsStepSize.y, supportSize.z)),
              d);
  d = OpUnion(FBoxCheap(railPosition -
                            vec3(0.0, StairsStepSize.y, -0.235 + supportSize.x),
                        vec3(supportSize.x, supportSize.x, 0.235)),
              d);

  return Hit(d, IdSafetyLine, position);
}
Hit Stairs(in vec3 position) {
  Hit rails = StairsRail(position);

  position.z -= PlatformLength();
  position.z += StairsStepSize.z * (14.0 * 0.5);
  position.y -= StairsStepSize.y * (12.0 * 0.5);

  vec3 boundingSize = (StairsStepSize * 0.5);
  boundingSize.yz *= 12.0;
  vec3 boundingPosition = position;
  boundingPosition.y -= StairsStepSize.y * 0.01;
  boundingPosition.z -= StairsStepSize.z;

  float d = FBoxCheap(boundingPosition, vec3(boundingSize * 1.5));

  if (d <= RayMarchingSurfaceDistance) {
    vec3 p = position;
    p.y /= StairsStepSize.y;
    p.z /= StairsStepSize.z;
    d = (min(p.y - floor(p.z), 1.0 - (p.z - floor(p.y))) * StairsStepSize.y *
         0.5);
  }

  d = OpIntersection(d, FBoxCheap(boundingPosition, vec3(boundingSize)));

  return OpUnionHit(Hit(d, IdStairs, boundingPosition), rails);
}

Hit Floor(in vec3 position) {
  float floorLength = PillarSpacing * (PillarCount - 1.0) + PillarSize.z * 0.5;
  float floorHeight = 5.0;
  position.y += floorHeight;

  Hit mainFloor =
      Hit(FBoxCheap2(position.xy, vec2(MainWidth * 0.5, floorHeight)),
          IdMainFloor, position);

  vec3 platformPosition = position;
  PMirror(platformPosition.x,
          (MainWidth + PlatformWidth) * 0.5 + SafetyLineSize);
  Hit platformFloor = Hit(
      FBoxCheap2(platformPosition.xy, vec2(PlatformWidth * 0.5, floorHeight)),
      IdPlatformFloor, platformPosition);

  vec3 safetyLinePosition = position;
  PMirror(safetyLinePosition.x, (MainWidth + SafetyLineSize) * 0.5);
  Hit safetyLineFloor = Hit(FBoxCheap2(safetyLinePosition.xy,
                                       vec2(SafetyLineSize * 0.5, floorHeight)),
                            IdSafetyLine, safetyLinePosition);

  Hit result = OpUnionHit(mainFloor, platformFloor);
  result = OpUnionHit(safetyLineFloor, result);

  return result;
}

vec3 SkyWindowDomain(in vec3 position) {
  vec3 q = position;
  q.z += PillarSpacing * 0.5;
  q.y -= PillarSize.y + BeamRadius + RoundRoofDepth;
  PModInterval1(q.z, PillarSpacing, 1.0, PillarCount - 1.0);
  return q;
}
float SkyWindowHoles(in vec3 position) {
  vec3 q = SkyWindowDomain(position);

  return FCylinder(q, SkyWindowSize.x, SkyWindowSize.y);
}
Hit SkyWindow(in vec3 position) {
  float skyWindowHoles = SkyWindowHoles(position);

  vec3 q = SkyWindowDomain(position);

  q.y -= SkyWindowSize.y;

  Hit inside = Hit(OpDifference(FCylinder(q, SkyWindowSize.x, SkyWindowSize.y),
                                skyWindowHoles + 0.01),
                   IdSkyWindow, q);

  Hit ring1 = Hit(
      OpDifference(FCylinder(q, SkyWindowSize.x + 0.10, SkyWindowSize.y + 0.15),
                   skyWindowHoles - 0.05),
      IdSkyWindowRingWhite, q);

  Hit ring2 = Hit(
      OpDifference(FCylinder(q, SkyWindowSize.x + 0.15, SkyWindowSize.y + 0.12),
                   skyWindowHoles + 0.01),
      IdSkyWindowRingBlack, q);

  Hit ring3 = Hit(
      OpDifference(FCylinder(q, SkyWindowSize.x + 0.20, SkyWindowSize.y + 0.08),
                   skyWindowHoles),
      IdSkyWindowRingWhite, q);

  Hit safetyRing = Hit(
      OpDifference(FCylinder(q, SkyWindowSize.x + 0.27, SkyWindowSize.y + 0.03),
                   skyWindowHoles),
      IdSafetyLine, q);

  Hit result;
  result = OpUnionHit(inside, ring1);
  result = OpUnionHit(ring2, result);
  result = OpUnionHit(ring3, result);
  result = OpUnionHit(safetyRing, result);

  return result;
}

Hit Roof(in vec3 position) {
  float skyWindowHoles = SkyWindowHoles(position);

  vec3 RoofSize = vec3(MainWidth * 0.5 + SafetyLineSize, 5.0, PlatformLength());
  vec3 roundRoofPosition = position;
  roundRoofPosition.y -= PillarSize.y + RoundRoofDepth + RoofSize.y;

  Hit roundRoof =
      Hit(OpDifference(
              OpDifference(
                  FBoxCheap(roundRoofPosition, RoofSize),
                  FBoxRound(roundRoofPosition +
                                vec3(0.0, RoofSize.y * 2.0 - BeamRadius, 0.0),
                            RoofSize * vec3(1.0, 1.0, 2.0), BeamRadius)),
              skyWindowHoles),
          IdRoundRoof, roundRoofPosition);

  vec3 platformRoofPosition = position;
  platformRoofPosition.y -= (PillarSize.y + RoofSize.y);
  PMirror(platformRoofPosition.x,
          (MainWidth + PlatformWidth) * 0.5 + SafetyLineSize);
  Hit platformRoof = Hit(
      FBoxCheap(platformRoofPosition, vec3(PlatformWidth * 0.5, RoofSize.yz)),
      IdPlatformRoof, platformRoofPosition);

  vec3 platformRoofLipPosition =
      platformRoofPosition - vec3(-PlatformWidth * 0.5 + BeamThickness - 0.01,
                                  -RoofSize.y + RoundRoofDepth * 0.25 - 0.01,
                                  0.0);
  vec3 roundRoofLipPosition =
      platformRoofPosition -
      vec3(-PlatformWidth * 0.5, -RoofSize.y + RoundRoofDepth, 0.0);

  Hit roofLip =
      Hit(OpUnion(FBoxRound(platformRoofLipPosition,
                            vec3(BeamThickness, RoundRoofDepth * 0.25,
                                 PlatformLength()),
                            0.02),
                  FBoxRound(roundRoofLipPosition,
                            vec3(BeamThickness * 2.0, BeamThickness * 0.25,
                                 PlatformLength()),
                            0.02)),
          IdPillarEdge, platformRoofPosition);

  Hit skyWindow = SkyWindow(position);

  Hit result;
  result = OpUnionHit(roundRoof, platformRoof);
  result = OpUnionHit(roofLip, result);
  result = OpUnionHit(skyWindow, result);

  return result;
}

Hit BackWall(in vec3 position) {
  const float wallDepth = 5.0;
  const vec3 wallSize = vec3((0.5 * MainWidth + SafetyLineSize + PlatformWidth),
                             PillarSize.y, wallDepth);

  position.z -= PlatformLength() + wallDepth;

  Hit wall = Hit(FBoxCheap(position, wallSize), IdPillarBody, position);

  vec3 edgePosition = position;
  edgePosition.z += wallSize.z;
  edgePosition.y -= wallSize.y;
  Hit edge = Hit(FBoxCheap(edgePosition, vec3(wallSize.x, 0.1, 0.01)),
                 IdPillarEdge, edgePosition);

  vec3 windowWallPosition = position - vec3(0.0, 5.0, 0.0);
  Hit windowWall =
      Hit(FBoxCheap(windowWallPosition, vec3(wallSize.x, 5.0, wallDepth - 1.0)),
          IdPillarBody, windowWallPosition);

  Hit result = OpUnionHit(wall, edge);
  result = OpUnionHit(windowWall, result);

  return result;
}

Hit BackGlass(in vec3 position) {
  float width = MainWidth * 0.5 + SafetyLineSize;
  float smallThickness = 0.01;
  float tileSize = width / 15.0;
  float separatorThickness = tileSize * 0.5;

  position.z -= PlatformLength();
  position.y -= PillarSize.y + 0.05;

  vec3 horizontalPosition = position;
  PModInterval1(horizontalPosition.y, tileSize, 0.0, 9.0);
  Hit horizontalLines =
      Hit(FBoxCheap(horizontalPosition,
                    vec3(width, smallThickness, smallThickness)),
          IdPillarEdge, horizontalPosition);

  vec3 verticalPosition = position;
  float verticalHeight = (BeamRadius + RoundRoofDepth) * 0.5;
  verticalPosition.y -= verticalHeight;
  PModInterval1(verticalPosition.x, tileSize, -15.0, 15.0);
  Hit verticalLines =
      Hit(FBoxCheap(verticalPosition,
                    vec3(smallThickness, verticalHeight, smallThickness)),
          IdPillarEdge, verticalPosition);

  vec3 separatorPosition = position;
  separatorPosition.y -= verticalHeight;
  separatorPosition.x -= tileSize * 3.5;
  PModInterval1(separatorPosition.x, tileSize * 7.0, -2.0, 1.0);
  Hit separator =
      Hit(FBoxCheap(separatorPosition,
                    vec3(separatorThickness, verticalHeight, smallThickness)),
          IdPillarEdge, separatorPosition);

  Hit result = OpUnionHit(verticalLines, horizontalLines);
  result = OpUnionHit(separator, result);

  return result;
}

Hit TrainTunnel(in vec3 position) {
  PMirror(position.x, 0.0);

  position.x -=
      (MainWidth + TrainTunnelSize.x) * 0.5 + SafetyLineSize + PlatformWidth;
  position.y -= PillarSize.y * 0.5;

  vec3 floorAndRoofPosition = position;

  PMirror(floorAndRoofPosition.y, TrainTunnelSize.y * 0.5);

  Hit floorAndRoof =
      Hit(FBoxCheap2(floorAndRoofPosition.xy - vec2(0.0, 1.0),
                     vec2(TrainTunnelSize.x * 0.5, 1.0)),
          IdTrainTunnel, floorAndRoofPosition - vec3(0.0, 1.0, 0.0));

  vec3 wallPosition = position;

  wallPosition.x -= TrainTunnelSize.x;

  Hit wall = Hit(FBoxCheap2(wallPosition.xy,
                            vec2(TrainTunnelSize.x * 0.5, TrainTunnelSize.y)),
                 IdTrainTunnelWall, wallPosition);

  Hit result = OpUnionHit(floorAndRoof, wall);

  return result;
}
Hit Train(in vec3 position) {
  const vec3 trainSize = vec3(TrainTunnelSize.xy * 0.375, 10.0);

  position.y -= PillarSize.y * 0.5;
  float side = PMirror(position.x, MainWidth * 0.5 + SafetyLineSize +
                                       PlatformWidth + TrainTunnelSize.x * 0.5);

  position.z -= mod(side * TIME * 44.0, 400.0) - 200.0;
  PModInterval1(position.z, trainSize.z * 2.0 + 0.5, -2.0, 2.0);

  float d =
      OpDifference(FBoxRound(position, trainSize, trainSize.y * 0.25),
                   FBoxRound(position, trainSize * 0.9, trainSize.y * 0.25));

  vec3 windowPosition = position - vec3(0.0, trainSize.y * 0.2, 0.0);
  PModInterval1(windowPosition.z, trainSize.z * 0.25, -2.0, 2.0);
  d = OpDifference(
      d, FBoxCheap(windowPosition, vec3(trainSize.x * 2.0, trainSize.y * 0.4,
                                        trainSize.z * 0.1)));

  Hit train = Hit(d, IdTrain, position);

  Hit result = train;

  return result;
}
Hit TrainSupportBeam(in vec3 position) {
  const float cylinderLength = TrainTunnelSize.y * 0.707;
  const float cylinderRadius = 0.125;

  PMod1(position.z, TrainTunnelSize.y);
  position.y -= PillarSize.y * 0.5;
  PRot(position.yz, PI / 4.0);
  PMirror(position.x,
          MainWidth * 0.5 + SafetyLineSize + PlatformWidth + TrainTunnelSize.x);

  float d = OpUnion(FCylinder(position, cylinderRadius, cylinderLength),
                    FCylinder(position.xzy, cylinderRadius, cylinderLength));

  return Hit(d, IdTrainSupportBeam, position);
}

Hit GetSceneData(in vec3 position) {
  vec3 p = position;

  Hit beam = Beam(p);
  Hit pillars = Pillars(p);
  Hit bench = Bench(p);
  Hit stairs = Stairs(p);
  Hit mainFloor = Floor(p);
  Hit roof = Roof(p);
  Hit backWall = BackWall(p);
  Hit backGlass = BackGlass(p);
  Hit trainTunnel = TrainTunnel(p);
  Hit train = Train(p);
  Hit trainSupportBeam = TrainSupportBeam(p);

  Hit result = beam;
  result = OpUnionHit(pillars, result);
  result = OpUnionHit(bench, result);
  result = OpUnionHit(stairs, result);
  result = OpUnionHit(mainFloor, result);
  result = OpUnionHit(roof, result);
  result = OpUnionHit(backWall, result);
  result = OpUnionHit(backGlass, result);
  result = OpUnionHit(trainTunnel, result);
  result = OpUnionHit(train, result);
  result = OpUnionHit(trainSupportBeam, result);

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
  case IdPillarBody:
    material.albedo = PillarBody.rgb;
    material.roughness = 0.95;
    material.metallic = 0.0;
    break;
  case IdPillarRed:
    material.albedo = PillarRed.rgb;
    material.roughness = 0.9;
    material.metallic = 0.0;
    break;
  case IdPillarGreen:
    material.albedo = PillarGreen.rgb;
    material.roughness = 0.9;
    material.metallic = 0.0;
    break;
  case IdPillarBottom:
    material.albedo = PillarBottom.rgb;
    material.roughness = 0.9;
    material.metallic = 0.0;
    break;
  case IdPillarPoster:
    material.albedo = PillarPoster.rgb;
    material.roughness = 0.9;
    material.metallic = 1.0;
    break;
  case IdPillarEdge:
    material.albedo = PillarEdge.rgb;
    material.roughness = 0.9;
    material.metallic = 1.0;
    break;
  case IdBench:
    material.albedo = vec3(0.5);
    material.roughness = 0.9;
    material.metallic = 1.0;
    break;
  case IdStairs:
    material.albedo = vec3(0.9);
    material.roughness = 0.95;
    material.metallic = 1.0;
    break;
  case IdMainFloor:
    material.albedo = vec3(0.8);
    material.roughness = 0.9;
    material.metallic = 0.0;
    break;
  case IdPlatformFloor:
    material.albedo = vec3(0.8);
    material.roughness = 0.9;
    material.metallic = 0.0;
    break;
  case IdSafetyLine:
    material.albedo = vec3(1.0, 0.3, 0.3);
    material.roughness = 0.95;
    material.metallic = 0.0;
    break;
  case IdPlatformRoof:
    material.albedo =
        mix(PlatformRoofColor.rgb, vec3(1.0, 0.3, 0.3), abs(normal.x));
    material.roughness = 0.95;
    material.metallic = 0.0;
    break;
  case IdRoundRoof:
    material.albedo = RoundRoofColor.rgb;
    material.roughness = 0.9;
    material.metallic = 0.0;
    break;
  case IdBeam:
    material.albedo = BeamColor.rgb;
    material.roughness = 0.9;
    material.metallic = 1.0;
    break;
  case IdSkyWindow:
    material.albedo = SkyWindowColor.rgb;
    material.roughness = 0.9;
    material.metallic = 0.0;
    break;
  case IdSkyWindowRingWhite:
    material.albedo = SkyWindowWhiteColor.rgb;
    material.roughness = 0.9;
    material.metallic = 0.0;
    break;
  case IdSkyWindowRingBlack:
    material.albedo = SkyWindowBlackColor.rgb;
    material.roughness = 0.9;
    material.metallic = 0.0;
    break;
  case IdTrainTunnel:
    material.albedo = SkyWindowBlackColor.rgb;
    material.roughness = 1.0;
    material.metallic = 0.0;
    break;
  case IdTrainTunnelWall: {
    vec3 squareColor = mix(TrainWallOrangeColor.rgb, TrainWallBlueColor.rgb,
                           dot(normal, vec3(-1.0, 0.0, 0.0)) * 0.5 + 0.5);
    squareColor = mix(squareColor, TrainWallGreenColor.rgb,
                      max(0.0, min(1.0, position.z / PlatformLength() * 0.25)));
    PRot(localPosition.yz, PI / 4.0);
    localPosition -= 0.5;
    vec2 squarePosition = floor(localPosition.zy);
    float brightness =
        cos(length(squarePosition * 1000.0 - vec2(TIME, 17.0))) *
            cos(length(squarePosition * 1000.0 - vec2(3.0, TIME * 1.1)) * 1.0) *
            0.1 +
        0.9;
    material.albedo = squareColor * brightness;
    material.roughness = 1.0;
    material.metallic = 0.0;
  } break;
  case IdTrainSupportBeam:
    material.albedo = TrainSupportBeamColor.rgb;
    material.roughness = 1.0;
    material.metallic = 0.0;
    break;
  case IdTrain:
    material.albedo = vec3(0.5) + 0.5 * cos(localPosition.z * 10.0) *
                                      sin(localPosition.y * 10.0);
    material.roughness = 1.0;
    material.metallic = 1.0;
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
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;

  vec3 linearColor = GetColor(uv);

  float ditherValue =
      DitherPattern[int(fragCoord.y) % 4][int(fragCoord.x) % 4] / 16.0;

  linearColor += (1.0 / 255.0) * ditherValue;

  vec3 color = LinearToGamma(linearColor);

  fragColor = vec4(color, 1.0);
}