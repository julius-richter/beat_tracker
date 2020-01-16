#include <cmath>
#include <torch/script.h>
#include <vector>

#include "../JuceLibraryCode/JuceHeader.h"
#include "BeatActivationComponent.h"
#include "Plot.h"
#include "utils.h"


BeatActivationComponent::BeatActivationComponent(std::vector<std::vector<float> >* filteredSpectrogram)
    : beatActivationImage (Image::RGB, 1000, 100, true)
{
    setOpaque (true);

    input = filteredSpectrogram;
    model = torch::jit::load("/Users/juliusrichter/Documents/Uni/Masterarbeit/beat_tracker/app/BeatTracker/Source/traced_model.pt");
}


BeatActivationComponent::~BeatActivationComponent() {}


void BeatActivationComponent::timerCallback() {}


void BeatActivationComponent::paint (Graphics& g) 
{
    g.fillAll (Colours::black);
    g.setOpacity (1.0f);
    g.drawImage (beatActivationImage, getLocalBounds().toFloat());

    if (activation.size() > 0)
    {
	    Graph* graph = new Graph(getLocalBounds(), "My Measurements", "Freq(Hz)", "db");
	    GraphDataset* leftEarData = new GraphDataset("Left", Colours::white);
	    for (unsigned int i = 0; i < activation.size(); ++i)
	    {
	        leftEarData->append(new GraphPoint(i, activation[i]));
	    }
	    graph->append(leftEarData);
	    graph->paint(g);

	    delete graph;
	}
}


float BeatActivationComponent::getValue()
{
    return 0;
}


std::vector<float> BeatActivationComponent::getVector()
{
    return activation;
}


void BeatActivationComponent::calculateBeatActivation(){
	long long frameSize = static_cast<long long>(input->size());
	long long freqBins = static_cast<long long>((*input)[0].size());
	std::vector<int64_t> sizes = {1, frameSize, freqBins};

	float contiguosData[frameSize][freqBins];
	for (int i = 0; i < frameSize; ++i)
	{
		for (int j = 0; j < freqBins; j++)
		{
			contiguosData[i][j] = (*input)[i][j];
		}
	}

    torch::Tensor in_tensor = torch::from_blob(&contiguosData, sizes);
    std::vector<torch::jit::IValue> inputs;
	inputs.push_back(in_tensor);
	torch::Tensor output = model.forward(inputs).toTensor();

	for (auto i = 0; i < output.sizes()[2]; ++i)
	{
		activation.push_back(exp(output[0][1][i].item<float>()));
	}
}




