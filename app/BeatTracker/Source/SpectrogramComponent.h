#ifndef SPECTROGRAM_COMPONENT_H
#define SPECTROGRAM_COMPONENT_H


class SpectrogramComponent : public Component, private Timer
{
public:
    SpectrogramComponent();

    ~SpectrogramComponent() override;

    void timerCallback() override;

    void paint (Graphics& g) override;

    void initialize(AudioSampleBuffer& fileBuffer);

    void calculateSTFT();
 
    void filterSpectogram();

    std::vector<std::vector<float> > createFilterband(int numFreqBin);

    std::vector<float> logFrequencies(int bandsPerOctavce, float fmin, float fmax);

    std::vector<float> calculateFftFrequencies(int windowLength, float sampleSpacing);

    std::vector<int> frequenciesToBins(std::vector<float> frequencies, std::vector<float> fftFrequencies);

    std::vector<std::vector<float> > binsToFilterbank(std::vector<int> bins, int numFreqBin);

    void generateSpectrogramImage();

    float getValue();

    std::string getText();

    enum
	{
	    fftOrder = 11, /* 2^11 = 2048 samples */
	    hopSize  = 411,
	    frameSize  = 1 << fftOrder,
	    numFreqBin = 1 << (fftOrder - 1)      
	};

    std::vector<std::vector<float> > filteredSpectogram;


private:
    dsp::FFT forwardFFT;
    std::vector<std::vector<float> > spectogram;
    Image spectrogramImage;

    std::string text;

    const float* signal; 

    int numSamples;
    int numFrames;
    float chunk[frameSize*2];
    // std::valarray<float> chunk(size_t frameSize);

    float maxLevel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SpectrogramComponent)
};

#endif