#pragma once


class BeatActivationComponent : public Component, private Timer
{
public:
	BeatActivationComponent(std::vector<std::vector<float> >* filteredSpectrogram);

	~BeatActivationComponent() override;

	void timerCallback() override;

    void paint (Graphics& g) override;

    float getValue();

	std::vector<float> getVector();

    void calculateBeatActivation();


private: 
    Image beatActivationImage;

    torch::jit::script::Module model;
    std::vector<std::vector<float> >* input;
    std::vector<float> activation;

	JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BeatActivationComponent)
};
