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

// TODO: trim this list of includes

#include <filament/Engine.h>
#include <filament/IndirectLight.h>
#include <filament/LightManager.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/TransformManager.h>
#include <filament/View.h>

#include <gltfio/Animator.h>
#include <gltfio/AssetLoader.h>
#include <gltfio/FilamentAsset.h>
#include <gltfio/ResourceLoader.h>

#include "app/Config.h"
#include "app/FilamentApp.h"
#include "app/IBL.h"

#include <getopt/getopt.h>

#include <imgui.h>

#include <utils/NameComponentManager.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

#include "generated/resources/resources.h"
#include "generated/resources/textures.h"

using namespace filament;
using namespace gltfio;
using namespace math;
using namespace utils;

#define SIDEBAR_WIDTH 350

struct App {
    Config config;
    Scene* scene;
    AssetLoader* loader;
    FilamentAsset* asset;
    Animator* animator;
    NameComponentManager* names;
    Entity sunlight;
    int currentAnimation = 1;
    bool resetAnimation = true;
    bool shadowPlane = false;
    float iblIntensity = 30000.0f;
    float iblRotation = 0.0f;
    bool showWireframe = false;
    bool addSunlight = true;
};

static const char* DEFAULT_IBL = "envs/venetian_crossroads";

static void printUsage(char* name) {
    std::string exec_name(Path(name).getName());
    std::string usage(
            "SHOWCASE renders the specified glTF file, or a built-in file if none is specified\n"
            "Usage:\n"
            "    SHOWCASE [options] <gltf file>\n"
            "Options:\n"
            "   --help, -h\n"
            "       Prints this message\n\n"
            "   --api, -a\n"
            "       Specify the backend API: opengl (default) or vulkan\n\n"
            "   --ibl=<path to cmgen IBL>, -i <path>\n"
            "       Override the built-in IBL\n\n"
            "   --shadow-plane, -p\n"
            "       Enable shadow plane\n\n"
    );
    const std::string from("SHOWCASE");
    for (size_t pos = usage.find(from); pos != std::string::npos; pos = usage.find(from, pos)) {
        usage.replace(pos, from.length(), exec_name);
    }
    std::cout << usage;
}

static int handleCommandLineArguments(int argc, char* argv[], App* app) {
    static constexpr const char* OPTSTR = "ha:pi:";
    static const struct option OPTIONS[] = {
            { "help",       no_argument,       nullptr, 'h' },
            { "api",        required_argument, nullptr, 'a' },
            { "ibl",        required_argument, nullptr, 'i' },
            { "shadow-plane", no_argument,     nullptr, 'p' },
            { nullptr, 0, nullptr, 0 }  // termination of the option list
    };
    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, OPTSTR, OPTIONS, &option_index)) >= 0) {
        std::string arg(optarg ? optarg : "");
        switch (opt) {
            default:
            case 'h':
                printUsage(argv[0]);
                exit(0);
            case 'a':
                if (arg == "opengl") {
                    app->config.backend = Engine::Backend::OPENGL;
                } else if (arg == "vulkan") {
                    app->config.backend = Engine::Backend::VULKAN;
                } else {
                    std::cerr << "Unrecognized backend. Must be 'opengl'|'vulkan'." << std::endl;
                }
                break;
            case 'i':
                app->config.iblDirectory = arg;
                break;
            case 'p':
                app->shadowPlane = true;
                break;
        }
    }
    return optind;
}

static std::ifstream::pos_type getFileSize(const char* filename) {
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

int main(int argc, char** argv) {
    App app;

    app.config.title = "showcase";
    app.config.iblDirectory = FilamentApp::getRootPath() + DEFAULT_IBL;

    int option_index = handleCommandLineArguments(argc, argv, &app);
    utils::Path filename;
    int num_args = argc - option_index;
    if (num_args >= 1) {
        filename = argv[option_index];
        if (!filename.exists()) {
            std::cerr << "file " << argv[option_index] << " not found!" << std::endl;
            return 1;
        }
    }

    auto setup = [&app, filename](Engine* engine, View* view, Scene* scene) {

        // We leave FXAA enabled but we also enable MSAA for a nice result. The wireframe looks
        // much better with MSAA enabled.
        view->setSampleCount(4);

        app.names = new NameComponentManager(EntityManager::get());
        app.loader = AssetLoader::create({engine, app.names, LOAD_UBERSHADERS});
        // app.loader = AssetLoader::create({engine, app.names, GENERATE_SHADERS});
        app.scene = scene;
        if (filename.isEmpty()) {
            return;
        }
        long contentSize = static_cast<long>(getFileSize(filename.c_str()));
        if (contentSize <= 0) {
            std::cerr << "Unable to open " << filename << std::endl;
            exit(1);
        }

        // Consume the glTF file.
        std::ifstream in(filename.c_str(), std::ifstream::in);
        std::vector<uint8_t> buffer(static_cast<unsigned long>(contentSize));
        if (!in.read((char*) buffer.data(), contentSize)) {
            std::cerr << "Unable to read " << filename << std::endl;
            exit(1);
        }

        // Parse the glTF file and create Filament entities.
        if (filename.getExtension() == "glb") {
            app.asset = app.loader->createAssetFromBinary(buffer.data(), buffer.size());
        } else {
            app.asset = app.loader->createAssetFromJson(buffer.data(), buffer.size());
        }
        if (!app.asset) {
            std::cerr << "Unable to parse " << filename << std::endl;
            exit(1);
        }
        buffer.clear();
        buffer.shrink_to_fit();

        // Compute the scale required to fit the model's bounding box into [-1, +1]
        auto minpt = app.asset->getBoundingBox().min;
        auto maxpt = app.asset->getBoundingBox().max;
        float maxExtent = 0;
        maxExtent = std::max(maxpt.x - minpt.x, maxpt.y - minpt.y);
        maxExtent = std::max(maxExtent, maxpt.z - minpt.z);
        float scaleFactor = 2.0f / maxExtent;
        float3 center = (minpt + maxpt) / 2.0f;
        center.z += 4.0f / scaleFactor;
        auto& tcm = engine->getTransformManager();
        auto root = tcm.getInstance(app.asset->getRoot());
        tcm.setTransform(root, mat4f::scaling(float3(scaleFactor)) * mat4f::translation(-center));

        // Load external textures and buffers.
        utils::Path assetFolder = filename.getParent();
        gltfio::ResourceLoader({engine, assetFolder, true, false}).loadResources(app.asset);

        // Load animation data then free the source hierarchy.
        app.animator = app.asset->getAnimator();
        app.asset->releaseSourceData();

        // Add renderables. This also adds transform-only entities that get ignored.
        scene->addEntities(app.asset->getEntities(), app.asset->getEntityCount());

        // Add a point light to enable shadows.
        app.sunlight = EntityManager::get().create();
        LightManager::Builder(LightManager::Type::SUN)
            .color(Color::toLinear<ACCURATE>({0.98, 0.92, 0.89}))
            .intensity(100000.0)
            .direction({0.6, -1.0, -0.8})
            .castShadows(true)
            .sunAngularRadius(1.9)
            .sunHaloSize(10.0)
            .sunHaloFalloff(80.0)
            .build(*engine, app.sunlight);
        if (app.addSunlight) {
            scene->addEntity(app.sunlight);
        }
    };

    auto cleanup = [&app](Engine* engine, View*, Scene*) {
        Fence::waitAndDestroy(engine->createFence());
        app.loader->destroyAsset(app.asset);
        app.loader->destroyMaterials();
        AssetLoader::destroy(&app.loader);
    };

    auto animate = [&app](Engine* engine, View* view, double now) {
        static double startTime = 0;
        const size_t numAnimations = app.animator->getAnimationCount();
        if (app.resetAnimation) {
            startTime = now;
            for (size_t i = 0; i < numAnimations; i++) {
                app.animator->applyAnimation(i, 0);
            }
            app.resetAnimation = false;
        }
        if (numAnimations > 0 && app.currentAnimation > 0) {
            app.animator->applyAnimation(app.currentAnimation - 1, now - startTime);
        }
        app.animator->updateBoneMatrices();
    };

    auto gui = [&app](filament::Engine* engine, filament::View* view) {

        auto& tm = engine->getTransformManager();
        auto& rm = engine->getRenderableManager();

        // Show a common set of UI widgets for all renderables.
        auto renderableTreeItem = [&app, &rm](utils::Entity entity) {
            bool rvis = app.scene->hasEntity(entity);
            ImGui::Checkbox("visible", &rvis);
            if (rvis) {
                app.scene->addEntity(entity);
            } else {
                app.scene->remove(entity);
            }
            auto instance = rm.getInstance(entity);
            size_t numPrims = rm.getPrimitiveCount(instance);
            for (size_t prim = 0; prim < numPrims; ++prim) {
                const Material* mat = rm.getMaterialInstanceAt(instance, prim)->getMaterial();
                const char* mname = mat->getName();
                if (mname) {
                    ImGui::Text("prim %zu: material %s", prim, mname);
                } else {
                    ImGui::Text("prim %zu: (unnamed material)", prim);
                }
            }
        };

        // Declare a std::function for tree nodes, it's an easy way to make a recursive lambda.
        std::function<void(utils::Entity)> treeNode;

        treeNode = [&](utils::Entity entity) {
            auto tinstance = tm.getInstance(entity);
            auto rinstance = rm.getInstance(entity);
            intptr_t treeNodeId = 1 + entity.getId();

            const char* label = rinstance ? "Mesh" : "Node";
            auto nameInstance = app.names->getInstance(entity);
            if (nameInstance) {
                label = app.names->getName(nameInstance);
            }

            ImGuiTreeNodeFlags flags = rinstance ? 0 : ImGuiTreeNodeFlags_DefaultOpen;
            std::vector<Entity> children(tm.getChildCount(tinstance));
            if (ImGui::TreeNodeEx((const void*) treeNodeId, flags, "%s", label)) {
                if (rinstance) {
                    renderableTreeItem(entity);
                }
                tm.getChildren(tinstance, children.data(), children.size());
                for (auto ce : children) {
                    treeNode(ce);
                }
                ImGui::TreePop();
            }
        };

        auto animationsTreeItem = [&]() {
            size_t count = app.animator->getAnimationCount();
            int selectedAnimation = app.currentAnimation;
            ImGui::RadioButton("Disable", &selectedAnimation, 0);
            for (size_t i = 0; i < count; ++i) {
                std::string label = app.animator->getAnimationName(i);
                if (label.empty()) {
                    label = "Unnamed " + std::to_string(i);
                }
                ImGui::RadioButton(label.c_str(), &selectedAnimation, i + 1);
            }
            if (selectedAnimation != app.currentAnimation) {
                app.currentAnimation = selectedAnimation;
                app.resetAnimation = true;
            }
        };

        // Disable rounding and draw a fixed-width ImGui window that looks like a sidebar.
        ImGui::GetStyle().WindowRounding = 0;
        ImGui::SetNextWindowPos(ImVec2(0, 0));

        const float width = ImGui::GetIO().DisplaySize.x;
        const float height = ImGui::GetIO().DisplaySize.y;
        ImGui::SetNextWindowSize(ImVec2(SIDEBAR_WIDTH, height), ImGuiCond_Once);
        ImGui::SetNextWindowSizeConstraints(ImVec2(20, height), ImVec2(width, height));

        ImGui::Begin("Filament", nullptr, ImGuiWindowFlags_NoTitleBar);
        if (ImGui::CollapsingHeader("Light")) {
            ImGui::SliderFloat("IBL intensity", &app.iblIntensity, 0.0f, 100000.0f);
            ImGui::SliderAngle("IBL rotation", &app.iblRotation);
            ImGui::Checkbox("Sunlight", &app.addSunlight);
            if (app.addSunlight) {
                app.scene->addEntity(app.sunlight);
            } else {
                app.scene->remove(app.sunlight);
            }
        }
        if (ImGui::CollapsingHeader("Model", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (app.animator->getAnimationCount() > 0) {
                intptr_t animationsNodeId = -1;
                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_DefaultOpen;
                if (ImGui::TreeNodeEx((const void*) animationsNodeId, flags, "Animations")) {
                    animationsTreeItem();
                    ImGui::TreePop();
                }
            }
            ImGui::Checkbox("Wireframe", &app.showWireframe);
            if (app.showWireframe) {
                app.scene->addEntity(app.asset->getWireframe());
            } else {
                app.scene->remove(app.asset->getWireframe());
            }
            treeNode(app.asset->getRoot());
        }
        FilamentApp::get().enableSidebar(ImGui::GetWindowWidth());
        ImGui::End();

        auto ibl = FilamentApp::get().getIBL()->getIndirectLight();
        ibl->setIntensity(app.iblIntensity);
        ibl->setRotation(mat3f::rotation(app.iblRotation, float3{ 0, 1, 0 }));
    };

    FilamentApp& filamentApp = FilamentApp::get();
    filamentApp.animate(animate);
    filamentApp.run(app.config, setup, cleanup, gui);

    delete app.names;

    return 0;
}
