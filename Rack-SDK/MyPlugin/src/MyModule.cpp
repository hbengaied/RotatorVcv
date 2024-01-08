/***********************************************************************
 * MyModule.cpp
 *
 * Auteur: Hicheme BEN GAIED
 * Contributeurs: Pierre COLLARD
 *
 * Ce fichier fait partie du plugin Rotator, utilisé pour créer
 * des modules audio dans l'environnement de VCV Rack. Il contient la
 * définition et l'implémentation de 'MyModule', un module de synthétiseur
 * modulaire personnalisé.
 *
 * Fonctionnalités Clés:
 * - Gestion de 1 à 5 entrées audio, permettant de traiter simultanément
 *   plusieurs sons.
 * - Fonctionnalité de rotation des sons entre les entrées, facilitant
 *   la création d'effets de spatialisation sonore en format 5.1.
 * - Utilisation de buffers circulaires pour le stockage et le traitement
 *   efficace des signaux audio.
 * - Intégration CUDA pour un traitement audio parallèle et performant sur GPU.
 * - Contrôle visuel à travers des lumières clignotantes pour refléter l'état
 *   et les actions du module.
 * - Intégration CUDA pour un traitement audio efficace et rapide sur GPU.
 * - Contrôle visuel avec des lumières clignotantes pour indiquer divers
 *   états et actions du module.
 ***********************************************************************/

#include <cmath>
#include <vector>
#include "plugin.hpp"

// class qui va me servir de buffer circulaire pour mes sons
class CircularBuffer
{
	std::vector<float> buffer;
	size_t index = 0;

public:
	CircularBuffer(size_t size) : buffer(size, 0.0f) {}

	// Constructeur par défaut
	CircularBuffer() : CircularBuffer(44100) {} // taille par defaut de 44100 de vcv

	// va me permettre d'ecrire dans mon buffer et d'incrementer l'index pour passer au buffer suivant
	void write(float sample)
	{
		buffer[index] = sample;
		index = (index + 1) % buffer.size();
	}

	// me permet de lire le buffer pour un outpurt voulu
	float read(size_t delay)
	{
		size_t readIndex = (index + buffer.size() - delay) % buffer.size();
		return buffer[readIndex];
	}
};

struct MyModule : Module
{
	float phase = 0.f;
	int blinkPhase = 0;
	float switchInterval = 6.0f;

	enum ParamId
	{
		PITCH_PARAM,
		PARAMS_LEN
	};
	enum InputId
	{
		INPUTONE_INPUT,
		INPUTTWO_INPUT,
		INPUTHREE_INPUT,
		INPUTFOUR_INPUT,
		INPUTFIVE_INPUT,
		INPUTS_LEN
	};
	enum OutputId
	{
		OUTPUTONE_OUTPUT,
		OUTPUTTWO_OUTPUT,
		OUTPUTTHREE_OUTPUT,
		OUTPUTFOUR_OUTPUT,
		OUTPUTFIVE_OUTPUT,
		OUTPUTS_LEN
	};
	enum LightId
	{
		BLINKONE_LIGHT,
		BLINKTWO_LIGHT,
		BLINKTHREE_LIGHT,
		BLINKFOUR_LIGHT,
		BLINKFIVE_LIGHT,
		LIGHTS_LEN
	};

	CircularBuffer inputBuffers[MyModule::INPUTS_LEN]; // un buffer pour chaque entree

	MyModule() : Module()
	{
		config(PARAMS_LEN, INPUTS_LEN, OUTPUTS_LEN, LIGHTS_LEN);
		configParam(PITCH_PARAM, -2.5f, 2.5f, 0.f, "");
		configInput(INPUTTWO_INPUT, "");
		configInput(INPUTONE_INPUT, "");
		configInput(INPUTHREE_INPUT, "");
		configInput(INPUTFIVE_INPUT, "");
		configInput(INPUTFOUR_INPUT, "");
		configOutput(OUTPUTTWO_OUTPUT, "");
		configOutput(OUTPUTONE_OUTPUT, "");
		configOutput(OUTPUTTHREE_OUTPUT, "");
		configOutput(OUTPUTFIVE_OUTPUT, "");
		configOutput(OUTPUTFOUR_OUTPUT, "");
	}

	void MyModule::process(const ProcessArgs &args) override
	{
		float pitchValue = params[PITCH_PARAM].getValue();
		int connectedOutputs = 0;
		std::vector<int> connectedOutputIndices;

		// check le nb d'outpub connecte a une sortie audio
		for (int i = 0; i < OUTPUTS_LEN; ++i)
		{
			if (outputs[i].isConnected())
			{
				connectedOutputs++;
				connectedOutputIndices.push_back(i);
			}
		}

		// eteindre toutes les lumières
		for (int i = 0; i < LIGHTS_LEN; ++i)
		{
			lights[i].setBrightness(0.f);
		}

		static float previousPitchValue = 0.f;
		static int sampleCounter = 0;
		static float crossfadeValue = 0.f;
		static bool isCrossfading = false;

		// Réinitialiser si pitchValue passe à 0
		if (pitchValue == 0.f && previousPitchValue != 0.f)
		{
			sampleCounter = 0;
			crossfadeValue = 0.f;
			isCrossfading = false;
			blinkPhase = 0;
		}
		previousPitchValue = pitchValue;

		// si pitchvalue ==0, alors les sons tournent pas ils sortent par leurs outputs respectifs
		if (pitchValue == 0.f)
		{
			for (int i = 0; i < INPUTS_LEN; ++i)
			{
				if (inputs[i].isConnected())
				{
					// outputs[i].setVoltage(inputs[i].getVoltage());
					float sample = inputs[i].getVoltage();
					inputBuffers[i].write(sample);
					lights[i].setBrightness(1.f);
				}
			}
		}
		else
		{
			switchInterval = 6.0f; // duree en sec avant chaque rota
			if (pitchValue > 0.f)
			{
				switchInterval -= (pitchValue * 2);
			}
			else if (pitchValue < 0.f)
			{
				switchInterval += (pitchValue * 2);
			}

			// me permet de savoir apres combien de signal audio la rotation doit se faire
			// et aussi la vitesse du crossafade
			const int samplesPerSwitch = static_cast<int>(args.sampleRate * switchInterval);
			const float crossfadeIncrement = 1.0f / samplesPerSwitch;

			// si on rentre dans le if ca veut dire qu'il y a au moins une sortie a traiter
			if (connectedOutputs > 0)
			{
				if (++sampleCounter >= samplesPerSwitch)
				{
					sampleCounter = 0;
					isCrossfading = true;
					// donne le sens de rota et incremente ou decremente la lumiere pour savoir dans quel sens je tourne
					if (pitchValue > 0)
					{
						blinkPhase = (blinkPhase + 1) % connectedOutputs;
					}
					else
					{
						blinkPhase = (blinkPhase - 1 + connectedOutputs) % connectedOutputs;
					}
				}

				// crossafe pour permettre une transition douce des sons entre les sorties -> conseil de pierre pcq il trouvait que le son se teleportait avant
				if (isCrossfading)
				{
					crossfadeValue += crossfadeIncrement;
					if (crossfadeValue >= 1.0f)
					{
						crossfadeValue = 0.f;
						isCrossfading = false;
					}
				}

				// je donne un poid egale pour le crossfade pour chaque inputs, ca va me permettre d'avoir un mixage des sons uniforme et pas un son qui domine l'autre
				float weights[INPUTS_LEN];
				for (int i = 0; i < INPUTS_LEN; ++i)
				{
					if (inputs[i].isConnected())
					{
						weights[i] = 1.0f / INPUTS_LEN;
					}
					else
					{
						weights[i] = 0.0f;
					}
				}

				// traitement cuda
				const int bufferSize = 128;
				float *deviceInputBuffers[INPUTS_LEN];
				float *deviceOutputBuffer;

				// allocation de mémoire gpu
				for (int i = 0; i < INPUTS_LEN; ++i)
				{
					cudaMalloc(&deviceInputBuffers[i], bufferSize * sizeof(float));
					// copie des data du buffer dentree cpuvers gpu
					cudaMemcpy(deviceInputBuffers[i], inputBuffers[i], bufferSize * sizeof(float), cudaMemcpyHostToDevice);
				}
				cudaMalloc(&deviceOutputBuffer, bufferSize * sizeof(float));

				// appel du kernel cuda
				runAudioProcessingKernel(deviceInputBuffers, deviceOutputBuffer, bufferSize, weights, INPUTS_LEN);

				// recup res
				float outputBuffer[bufferSize];
				cudaMemcpy(outputBuffer, deviceOutputBuffer, bufferSize * sizeof(float), cudaMemcpyDeviceToHost);

				// traitement des sorties
				for (int i = 0; i < connectedOutputs; ++i)
				{
					int outputIndex = connectedOutputIndices[i];

					// utilisation des donnees traitees par CUDA pour chaque sortie
					for (int j = 0; j < bufferSize; ++j)
					{
						outputs[outputIndex].setVoltage(outputBuffer[j]);
					}

					// allume la lumiere correspondante
					lights[outputIndex].setBrightness(1.f);
				}

				// free de la mémoire GPU
				for (int i = 0; i < INPUTS_LEN; ++i)
				{
					cudaFree(deviceInputBuffers[i]);
				}
				cudaFree(deviceOutputBuffer);
			}
		}
	}
};

struct MyModuleWidget : ModuleWidget
{
	MyModuleWidget(MyModule *module)
	{
		setModule(module);
		setPanel(createPanel(asset::plugin(pluginInstance, "res/MyModule.svg")));

		addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));
		addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));

		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(44.64, 111.831)), module, MyModule::PITCH_PARAM));

		addInput(createInputCentered<PJ301MPort>(mm2px(Vec(44.64, 44.33)), module, MyModule::INPUTTWO_INPUT));
		addInput(createInputCentered<PJ301MPort>(mm2px(Vec(35.084, 54.988)), module, MyModule::INPUTONE_INPUT));
		addInput(createInputCentered<PJ301MPort>(mm2px(Vec(53.904, 55.177)), module, MyModule::INPUTHREE_INPUT));
		addInput(createInputCentered<PJ301MPort>(mm2px(Vec(35.084, 74.698)), module, MyModule::INPUTFIVE_INPUT));
		addInput(createInputCentered<PJ301MPort>(mm2px(Vec(53.904, 74.698)), module, MyModule::INPUTFOUR_INPUT));

		addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(44.64, 19.081)), module, MyModule::OUTPUTTWO_OUTPUT));
		addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(16.98, 41.835)), module, MyModule::OUTPUTONE_OUTPUT));
		addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(72.3, 41.835)), module, MyModule::OUTPUTTHREE_OUTPUT));
		addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(13.525, 91.69)), module, MyModule::OUTPUTFIVE_OUTPUT));
		addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(69.276, 91.69)), module, MyModule::OUTPUTFOUR_OUTPUT));

		addChild(createLightCentered<MediumLight<RedLight>>(mm2px(Vec(44.64, 34.125)), module, MyModule::BLINKTWO_LIGHT));
		addChild(createLightCentered<MediumLight<RedLight>>(mm2px(Vec(62.133, 49.433)), module, MyModule::BLINKTHREE_LIGHT));
		addChild(createLightCentered<MediumLight<RedLight>>(mm2px(Vec(25.634, 50.0)), module, MyModule::BLINKONE_LIGHT));
		addChild(createLightCentered<MediumLight<RedLight>>(mm2px(Vec(21.855, 83.262)), module, MyModule::BLINKFIVE_LIGHT));
		addChild(createLightCentered<MediumLight<RedLight>>(mm2px(Vec(62.487, 83.262)), module, MyModule::BLINKFOUR_LIGHT));
	}
};

Model *modelMyModule = createModel<MyModule, MyModuleWidget>("MyModule");
