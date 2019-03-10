/*
 * Copyright (C) 2019 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gltfio/MaterialProvider.h>

#include <filamat/MaterialBuilder.h> // TODO: remove

#include <filament/MaterialInstance.h>
#include <filament/Texture.h>
#include <filament/TextureSampler.h>

#include <math/mat4.h>

#include <utils/Log.h>
#include <utils/Hash.h>

#include <tsl/robin_map.h>

#include <string>

#include "gltfresources.h"

using namespace filamat;
using namespace filament;
using namespace filament::math;
using namespace gltfio;
using namespace utils;

namespace {

class UbershaderLoader : public MaterialProvider {
public:
    UbershaderLoader(filament::Engine* engine);
    ~UbershaderLoader();

    filament::MaterialInstance* createMaterialInstance(MaterialKey* config, UvMap* uvmap,
            const char* label) override;

    size_t getMaterialsCount() const noexcept override;
    const filament::Material* const* getMaterials() const noexcept override;
    void destroyMaterials() override;

    using HashFn = utils::hash::MurmurHashFn<MaterialKey>;
    tsl::robin_map<MaterialKey, filament::Material*, HashFn> mCache;
    Material* mMaterial;
    std::vector<Material*> mMaterials;
    filament::Engine* mEngine;
};

UbershaderLoader::UbershaderLoader(Engine* engine) : mEngine(engine) {
    MaterialBuilder::init();
    mMaterial = Material::Builder()
        .package(GLTFRESOURCES_UBERSHADER_DATA, GLTFRESOURCES_UBERSHADER_SIZE)
        .build(*engine);
}

UbershaderLoader::~UbershaderLoader() {
    MaterialBuilder::shutdown();
}

size_t UbershaderLoader::getMaterialsCount() const noexcept {
    return mMaterials.size();
}

const Material* const* UbershaderLoader::getMaterials() const noexcept {
    return mMaterials.data();
}

void UbershaderLoader::destroyMaterials() {
    for (auto& iter : mCache) {
        mEngine->destroy(iter.second);
    }
    mMaterials.clear();
    mCache.clear();
}

static std::string shaderFromKey(const MaterialKey& config) {
    return R"SHADER(
        void material(inout MaterialInputs material) {
            float2 uvs[2] = { getUV0(), getUV1() };
            #if !defined(SHADING_MODEL_UNLIT)
                if (materialParams.normalIndex > -1) {
                    float2 uv = uvs[materialParams.normalIndex];
                    uv = (vec3(uv, 1.0) * materialParams.normalUvMatrix).xy;
                    material.normal = texture(materialParams_normalMap, uv).xyz * 2.0 - 1.0;
                    material.normal.y = -material.normal.y;
                    material.normal.xy *= materialParams.normalScale;
                }
            #endif
            prepareMaterial(material);
            material.baseColor = materialParams.baseColorFactor;
            if (materialParams.baseColorIndex > -1) {
                float2 uv = uvs[materialParams.baseColorIndex];
                uv = (vec3(uv, 1.0) * materialParams.baseColorUvMatrix).xy;
                material.baseColor *= texture(materialParams_baseColorMap, uv);
            }

            if (materialParams.blendEnabled) {
                material.baseColor.rgb *= material.baseColor.a;
            }

            material.baseColor *= getColor();

            #if !defined(SHADING_MODEL_UNLIT)
                material.roughness = materialParams.roughnessFactor;
                material.metallic = materialParams.metallicFactor;
                material.emissive.rgb = materialParams.emissiveFactor.rgb;
                material.emissive.a = 3.0;
                if (materialParams.metallicRoughnessIndex > -1) {
                    float2 uv = uvs[materialParams.metallicRoughnessIndex];
                    uv = (vec3(uv, 1.0) * materialParams.metallicRoughnessUvMatrix).xy;
                    vec4 roughness = texture(materialParams_metallicRoughnessMap, uv);
                    material.roughness *= roughness.g;
                    material.metallic *= roughness.b;
                }
                if (materialParams.aoIndex > -1) {
                    float2 uv = uvs[materialParams.aoIndex];
                    uv = (vec3(uv, 1.0) * materialParams.occlusionUvMatrix).xy;
                    material.ambientOcclusion = texture(materialParams_occlusionMap, uv).r *
                            materialParams.aoStrength;
                }
                if (materialParams.emissiveIndex > -1) {
                    float2 uv = uvs[materialParams.emissiveIndex];
                    uv = (vec3(uv, 1.0) * materialParams.emissiveUvMatrix).xy;
                    material.emissive.rgb *= texture(materialParams_emissiveMap, uv).rgb;
                }
            #endif
        }
    )SHADER";
}

static Material* createMaterial(Engine* engine, const MaterialKey& config, const UvMap& uvmap,
        const char* name) {
    using CullingMode = MaterialBuilder::CullingMode;
    std::string shader = shaderFromKey(config);
    MaterialBuilder builder = MaterialBuilder()
            .name(name)
            .flipUV(false)
            .material(shader.c_str())
            .doubleSided(config.doubleSided);

    static_assert(std::tuple_size<UvMap>::value == 8, "Badly sized uvset.");
    int numTextures = std::max({
        uvmap[0], uvmap[1], uvmap[2], uvmap[3],
        uvmap[4], uvmap[5], uvmap[6], uvmap[7],
    });
    builder.require(VertexAttribute::UV0);
    builder.require(VertexAttribute::UV1);
    builder.require(VertexAttribute::COLOR);

    // BASE COLOR
    builder.parameter(MaterialBuilder::UniformType::INT, "baseColorIndex");
    builder.parameter(MaterialBuilder::UniformType::FLOAT4, "baseColorFactor");
    builder.parameter(MaterialBuilder::SamplerType::SAMPLER_2D, "baseColorMap");
    builder.parameter(MaterialBuilder::UniformType::MAT3, "baseColorUvMatrix");
    builder.parameter(MaterialBuilder::UniformType::BOOL, "blendEnabled");

    // METALLIC-ROUGHNESS
    builder.parameter(MaterialBuilder::UniformType::INT, "metallicRoughnessIndex");
    builder.parameter(MaterialBuilder::UniformType::FLOAT, "metallicFactor");
    builder.parameter(MaterialBuilder::UniformType::FLOAT, "roughnessFactor");
    builder.parameter(MaterialBuilder::SamplerType::SAMPLER_2D, "metallicRoughnessMap");
    builder.parameter(MaterialBuilder::UniformType::MAT3, "metallicRoughnessUvMatrix");

    // NORMAL MAP
    builder.parameter(MaterialBuilder::UniformType::INT, "normalIndex");
    builder.parameter(MaterialBuilder::UniformType::FLOAT, "normalScale");
    builder.parameter(MaterialBuilder::SamplerType::SAMPLER_2D, "normalMap");
    builder.parameter(MaterialBuilder::UniformType::MAT3, "normalUvMatrix");

    // AMBIENT OCCLUSION
    builder.parameter(MaterialBuilder::UniformType::INT, "aoIndex");
    builder.parameter(MaterialBuilder::UniformType::FLOAT, "aoStrength");
    builder.parameter(MaterialBuilder::SamplerType::SAMPLER_2D, "occlusionMap");
    builder.parameter(MaterialBuilder::UniformType::MAT3, "occlusionUvMatrix");

    // EMISSIVE
    builder.parameter(MaterialBuilder::UniformType::INT, "emissiveIndex");
    builder.parameter(MaterialBuilder::UniformType::FLOAT3, "emissiveFactor");
    builder.parameter(MaterialBuilder::SamplerType::SAMPLER_2D, "emissiveMap");
    builder.parameter(MaterialBuilder::UniformType::MAT3, "emissiveUvMatrix");

    switch(config.alphaMode) {
        case AlphaMode::OPAQUE:
            builder.blending(MaterialBuilder::BlendingMode::OPAQUE);
            break;
        case AlphaMode::MASK:
            builder.blending(MaterialBuilder::BlendingMode::MASKED);
            builder.maskThreshold(config.alphaMaskThreshold);
            break;
        case AlphaMode::BLEND:
            builder.blending(MaterialBuilder::BlendingMode::TRANSPARENT);
            builder.depthWrite(true);
    }

    builder.shading(config.unlit ? Shading::UNLIT : Shading::LIT);

    Package pkg = builder.build();
    return Material::Builder().package(pkg.getData(), pkg.getSize()).build(*engine);
}

MaterialInstance* UbershaderLoader::createMaterialInstance(MaterialKey* config, UvMap* uvmap,
        const char* label) {
    gltfio::details::constrainMaterial(config, uvmap);
    auto iter = mCache.find(*config);
    TextureSampler sampler;
    auto getUvIndex = [uvmap](uint8_t srcIndex, bool hasTexture) -> int {
        return hasTexture ? int(uvmap->at(srcIndex)) - 1 : -1;
    };
    Material* material;
    if (iter == mCache.end()) {
        material = createMaterial(mEngine, *config, *uvmap, label);
        mCache.emplace(std::make_pair(*config, material));
        mMaterials.push_back(material);
    } else {
        material = iter->second;
    }
    mat3f identity;
    MaterialInstance* mi = material->createInstance();
    mi->setParameter("baseColorIndex",
            getUvIndex(config->baseColorUV, config->hasBaseColorTexture));
    mi->setParameter("normalIndex", getUvIndex(config->normalUV, config->hasNormalTexture));
    mi->setParameter("metallicRoughnessIndex",
            getUvIndex(config->metallicRoughnessUV, config->hasMetallicRoughnessTexture));
    mi->setParameter("aoIndex", getUvIndex(config->aoUV, config->hasOcclusionTexture));
    mi->setParameter("emissiveIndex", getUvIndex(config->emissiveUV, config->hasEmissiveTexture));
    mi->setParameter("baseColorUvMatrix", identity);
    mi->setParameter("metallicRoughnessUvMatrix", identity);
    mi->setParameter("normalUvMatrix", identity);
    mi->setParameter("occlusionUvMatrix", identity);
    mi->setParameter("emissiveUvMatrix", identity);
    mi->setParameter("blendEnabled", config->alphaMode == AlphaMode::BLEND);
    return mi;
}

} // anonymous namespace

namespace gltfio {

MaterialProvider* MaterialProvider::createUbershaderLoader(filament::Engine* engine) {
    return new UbershaderLoader(engine);
}

} // namespace gltfio
