#ifndef BEAT_ACTIVATION_COMPONENT_H
#define BEAT_ACTIVATION_COMPONENT_H


class BeatActivationComponent : public Component, private Timer
{
public:
	BeatActivationComponent(std::vector<std::vector<float> >* filteredSpectrogram);

	~BeatActivationComponent() override;

	void timerCallback() override;

    void paint (Graphics& g) override;

    void calculateBeatActivation();

    std::vector<double> activations;


private: 
    Image beatActivationImage;

    torch::jit::script::Module model;
    std::vector<std::vector<float> > *input;

	JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BeatActivationComponent)
};


#endif