#pragma once


class FilteredSpectrogramComponent : public Component, private Timer
{
public:
    FilteredSpectrogramComponent(std::vector<std::vector<float> >& spectogram)    
    : filteredSpectogramImage (Image::RGB, 1000, 81, true),
      spectogram(spectogram)
    {
        setOpaque (true);
    }

    ~FilteredSpectrogramComponent() override
    {
    }

    enum
    {
        fftOrder = 11,     /* 2^11 = 2048 samples */
        hopSize  = 411,
        frameSize  = 1 << fftOrder,
        numFreqBin = 1 << (fftOrder - 1)      
    };

    void initialize(AudioSampleBuffer& fileBuffer)
    {
        numSamples = fileBuffer.getNumSamples();
        numFrames = (int) std::floor(((float)numSamples - (float)frameSize) / (float)hopSize + 1.0);
    }


    void paint (Graphics& g) override
    {
        g.fillAll (Colours::black);
        g.setOpacity (1.0f);
        g.drawImage (filteredSpectogramImage, getLocalBounds().toFloat());
    }

    void timerCallback() override {}

    void filterSpectogram()
    {
        std::vector<std::vector<float> > filterbank = createFilterband(numFreqBin);
        std::vector<std::vector<float> > filt(numFrames, std::vector<float>(filterbank.size(), 0.0));

        for (auto t = 0 ; t < numFrames ; ++t)
        {
            for (unsigned long int m = 0 ; m < filterbank.size() ; ++m)
            {   
                filt[t][m] = 0;
                for (auto f = 0 ; f < numFreqBin ; ++f)
                    filt[t][m] += filterbank[m][f] * spectogram[t][f];
            }
        }

        for (auto &row : filt) 
            for (auto &elem : row) 
            {  
                elem = log(elem + 1);
            }

        maxLevel = std::numeric_limits<float>::lowest();

        for (const auto &v : filt)
        {   
            double current_max = *std::max_element(v.cbegin(), v.cend());
            maxLevel = maxLevel < current_max ? current_max : maxLevel;
        }

        filteredSpectogram = filt;
    }

    std::vector<std::vector<float> > createFilterband(int numFreqBin)
    {
        const int bandsPerOctavce = 12;
        const float fmin = 30.0f; 
        const float fmax = 17000.0f;

        std::vector<float> frequencies = logFrequencies(bandsPerOctavce, fmin, fmax);
        std::vector<float> fftFrequencies = calculateFftFrequencies(frameSize, 1.0f / 44100.0f);
        std::vector<int> bins = frequenciesToBins(frequencies, fftFrequencies);
        std::vector<std::vector<float> > filterbank = binsToFilterbank(bins, numFreqBin); 
        return filterbank;
    }

    std::vector<float> logFrequencies(int bandsPerOctavce, float fmin, float fmax)
    {
        const float fref = 440.0f;

        int left = (int) std::floor(log2(fmin / fref) * (float) bandsPerOctavce);
        int right = (int) std::ceil(log2(fmax / fref) * (float) bandsPerOctavce);
        int numBins = right - left;

        std::vector<int> freqRange = range(left, right);
        std::vector<float> frequencies(numBins); 

        for (int i = 0 ; i < numBins; ++i)
        {
            frequencies[i] = fref * pow(2.0, (float) freqRange[i] / (float) bandsPerOctavce); 
        }

        for (int i = 0 ; i < numBins; ++i)
        {
            if (frequencies[i] < fmin || frequencies[i] > fmax)
                frequencies.erase (frequencies.begin()+i);
        } 
        return frequencies;
    } 


    std::vector<float> calculateFftFrequencies(int windowLength, float sampleSpacing)
    {
        float val = 1.0 / ((float) windowLength * sampleSpacing);
        float N = (std::floor((windowLength - 1) / 2) + 1);
        std::vector<float> v = range(0.0f, N);
        std::transform(v.begin(), v.end(), v.begin(), [&val](auto& c){return c*val;});
        return v;
    }


    std::vector<int> frequenciesToBins(std::vector<float> frequencies, std::vector<float> fftFrequencies)
    {
        std::vector<int> indices;
        std::vector<float>::iterator idx;
        std::vector<float> left;
        std::vector<float> right;

        for (unsigned long int i = 0 ; i < frequencies.size() ; ++i)
        {
            idx = std::lower_bound(fftFrequencies.begin(), fftFrequencies.end(), frequencies[i]);
            indices.push_back((int) (idx - fftFrequencies.begin()));
        }
        
        for (auto &i : indices)
        {
            left.push_back(fftFrequencies[i-1]);
            right.push_back(fftFrequencies[i]);
        }

        for (unsigned long int i = 0 ; i < frequencies.size() ; ++i)
        {
            indices[i] = indices[i] - (int) (frequencies[i] - left[i] < right[i] - frequencies[i]);
        }

        return indices;
    }


    std::vector<std::vector<float> > binsToFilterbank(std::vector<int> bins, int numFreqBin)
    {
        std::vector<std::vector<float> > filterbank;
        int start;
        int center;
        int end;
        float sum_of_elems;

        for (unsigned long int i = 0 ; i < bins.size() - 2 ; ++i)
        {
            start = bins[i];
            center = bins[i+1];
            end = bins[i+2];
            std::vector<float> filt(numFreqBin, 0.0);

            for (auto t = start; t < center; ++t)
            {
                float delta = 1.0f / (float) (center - start);
                filt[t] = delta * (float) (t - start);
            }

            for (auto t = center; t < end; ++t)
            {
                float delta = 1.0f / (float) (end - center); 
                filt[t] = 1.0f - delta * (float) (t - center);  
            }

            sum_of_elems = 0.0f;

            for (auto& n : filt) 
                sum_of_elems += n;

            if (sum_of_elems > 0.0f)
                filterbank.push_back(filt);

        }
        return filterbank;
    }

    void generateFilteredSpectrogramImage()
    {
        auto imageHeight = filteredSpectogramImage.getHeight();
        auto imageWidth = filteredSpectogramImage.getWidth();

        for (int x = 0 ; x < imageWidth; ++x)
        {
            for (int y = 0; y < imageHeight; ++y)
            {
                // auto skewedProportionY = 1.0f - std::exp (std::log (y / (float) imageHeight) * 0.2f);
                // auto fftDataIndexY = jlimit (0, (int) numFreqBin, (int) (skewedProportionY * numFreqBin));

                auto fffDataIndexX = (int) ((numFrames - 1) * x / imageWidth);

                float level = filteredSpectogram[fffDataIndexX][y] / maxLevel;
                filteredSpectogramImage.setPixelAt (x, (imageHeight - 1) - y, Colour::fromHSV (level, 1.0f, level, 1.0f));      
            }
        }
    }

    auto getValue()
    {
        return maxLevel;
    }

private:
    Image filteredSpectogramImage;

    int numSamples;
    int numFrames;
    float maxLevel;

    std::vector<std::vector<float> >& spectogram;
    std::vector<std::vector<float> > filteredSpectogram;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (FilteredSpectrogramComponent)
};



class SpectrogramComponent : public Component, private Timer
{
public:
    SpectrogramComponent()
        : forwardFFT (fftOrder),
          spectrogramImage (Image::RGB, 1000, numFreqBin, true)
    {
        setOpaque (true);
    }

    ~SpectrogramComponent() override
    {
    }

    void initialize(AudioSampleBuffer& fileBuffer)
    {
        signal = fileBuffer.getReadPointer(0);
        numSamples = fileBuffer.getNumSamples();
        numFrames = (int) std::floor(((float)numSamples - (float)frameSize) / (float)hopSize + 1.0);
    }

    void calculateSTFT()
    {
        for (int i = 0; i < numFrames; ++i)
        {

            zeromem (chunk, sizeof chunk);
            memcpy(&chunk, signal, sizeof chunk);

            forwardFFT.performFrequencyOnlyForwardTransform (chunk); 

            std::vector<float> freqBins(chunk, chunk + frameSize);

            for (auto &elem : freqBins) 
                {  
                    elem = elem*elem;
                }

            spectogram.push_back(freqBins);
            
            signal += hopSize;
        }

        maxLevel = std::numeric_limits<float>::lowest();
        for (const auto &v : spectogram)
        {   
            double current_max = *std::max_element(v.cbegin(), v.cend());
            maxLevel = maxLevel < current_max ? current_max : maxLevel;
        }
    }

    void paint (Graphics& g) override
    {
        g.fillAll (Colours::black);
        g.setOpacity (1.0f);
        g.drawImage (spectrogramImage, getLocalBounds().toFloat());
    }

    void generateSpectrogramImage()
    {
        auto imageHeight = spectrogramImage.getHeight();
        auto imageWidth = spectrogramImage.getWidth();

        for (int x = 0 ; x < imageWidth; ++x)
        {
            for (int y = 0; y < imageHeight; ++y)
            {
                auto skewedProportionY = 1.0f - std::exp (std::log (y / (float) imageHeight) * 0.2f);
                auto fftDataIndexY = jlimit (0, (int) numFreqBin, (int) (skewedProportionY * numFreqBin));
                auto fffDataIndexX = (int) ((numFrames - 1) * x / imageWidth);

                float level = log(spectogram[fffDataIndexX][fftDataIndexY] + 1) / log(maxLevel);
                spectrogramImage.setPixelAt (x, y, Colour::fromHSV (level, 1.0f, level, 1.0f));      
            }
        }
    }

    auto getValue()
    {
        return 0;
    }

    void timerCallback() override {}

    enum
    {
        fftOrder = 11,     /* 2^11 = 2048 samples */
        hopSize  = 411,
        frameSize  = 1 << fftOrder,
        numFreqBin = 1 << (fftOrder - 1)      
    };

    std::vector<std::vector<float> > spectogram;

private:
    dsp::FFT forwardFFT;
    Image spectrogramImage;

    const float* signal; 

    int numSamples;
    int numFrames;
    float chunk[frameSize*2];
    float maxLevel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SpectrogramComponent)
};


class SimpleThumbnailComponent : public Component,
                                 private ChangeListener
{
public:
    SimpleThumbnailComponent (int sourceSamplesPerThumbnailSample,
                              AudioFormatManager& formatManager,
                              AudioThumbnailCache& cache)
       : thumbnail (sourceSamplesPerThumbnailSample, formatManager, cache)
    {
        thumbnail.addChangeListener (this);
    }

    void setFile (const File& file)
    {
        thumbnail.setSource (new FileInputSource (file));
    }

    void paint (Graphics& g) override
    {
        if (thumbnail.getNumChannels() == 0)
            paintIfNoFileLoaded (g);
        else
            paintIfFileLoaded (g);
    }

    void paintIfNoFileLoaded (Graphics& g)
    {
        g.fillAll (Colours::white);
        g.setColour (Colours::black);
        g.drawFittedText ("No File Loaded", getLocalBounds(), Justification::centred, 1);
    }

    void paintIfFileLoaded (Graphics& g)
    {
        g.fillAll(Colours::white);

        g.setColour (Colours::blue);
        thumbnail.drawChannels (g, getLocalBounds(), 0.0, thumbnail.getTotalLength(), 1.0f);
    }

    void changeListenerCallback (ChangeBroadcaster* source) override
    {
        if (source == &thumbnail)
            thumbnailChanged();
    }

private:
    void thumbnailChanged()
    {
        repaint();
    }

    AudioThumbnail thumbnail;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SimpleThumbnailComponent)
};

//------------------------------------------------------------------------------

class SimplePositionOverlay : public Component,
                              private Timer
{
public:
    SimplePositionOverlay (AudioTransportSource& transportSourceToUse)
       : transportSource (transportSourceToUse)
    {
        startTimer (40);
    }

    void paint (Graphics& g) override
    {
        auto duration = (float) transportSource.getLengthInSeconds();

        if (duration > 0.0)
        {
            auto audioPosition = (float) transportSource.getCurrentPosition();
            auto drawPosition = (audioPosition / duration) * getWidth();

            g.setColour (Colours::green);
            g.drawLine (drawPosition, 0.0f, drawPosition, (float) getHeight(), 2.0f);
        }
    }

    void mouseDown (const MouseEvent& event) override
    {
        auto duration = transportSource.getLengthInSeconds();

        if (duration > 0.0)
        {
            auto clickPosition = event.position.x;
            auto audioPosition = (clickPosition / getWidth()) * duration;

            transportSource.setPosition (audioPosition);
        }
    }

private:
    void timerCallback() override
    {
        repaint();
    }

    AudioTransportSource& transportSource;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SimplePositionOverlay)
};

//==============================================================================

class MainContentComponent: public AudioAppComponent,
                            public ChangeListener,
                            public Timer
{
public:
    MainContentComponent()
      : state (Stopped),
        thumbnailCache (5), 
        thumbnailComp (512, formatManager, thumbnailCache),
        positionOverlay (transportSource),
        filteredSpectrogramComp (spectrogramComp.spectogram)
    {
        addAndMakeVisible (&openButton);
        openButton.setButtonText ("Open...");
        openButton.onClick = [this] { openButtonClicked(); };

        addAndMakeVisible (&playButton);
        playButton.setButtonText ("Play");
        playButton.onClick = [this] { playButtonClicked(); };
        playButton.setColour (TextButton::buttonColourId, Colours::green);
        playButton.setEnabled (false);

        addAndMakeVisible (&stopButton);
        stopButton.setButtonText ("Stop");
        stopButton.onClick = [this] { stopButtonClicked(); };
        stopButton.setColour (TextButton::buttonColourId, Colours::red);
        stopButton.setEnabled (false);

        addAndMakeVisible (&currentPositionLabel);
        currentPositionLabel.setText ("00:00:000", dontSendNotification);

        addAndMakeVisible (&textButton);
        textButton.setButtonText ("Text");
        textButton.onClick = [this] { textButtonClicked(); };  

        addAndMakeVisible (&processButton);
        processButton.setButtonText ("Process");
        processButton.onClick = [this] { processButtonClicked(); };  
        processButton.setEnabled (false);

        addAndMakeVisible (&thumbnailComp);
        addAndMakeVisible (&positionOverlay);     

        addAndMakeVisible (infoText);
        infoText.setColour (Label::backgroundColourId, Colours::black);

        addAndMakeVisible (&filteredSpectrogramComp);

        setSize (600, 500);

        formatManager.registerBasicFormats();
        transportSource.addChangeListener (this);

        setAudioChannels (2, 2);
        startTimer (20);

        module = torch::jit::load("/Users/juliusrichter/Documents/Uni/Masterarbeit/beat_tracker/app/BeatTracker/Source/traced_model.pt");
    }

    ~MainContentComponent() override
    {
        shutdownAudio();
    }

    void prepareToPlay (int samplesPerBlockExpected, double sampleRate) override
    {
        transportSource.prepareToPlay (samplesPerBlockExpected, sampleRate);
    }

    void getNextAudioBlock (const AudioSourceChannelInfo& bufferToFill) override
    {
        if (readerSource.get() == nullptr)
        {
            bufferToFill.clearActiveBufferRegion();
            return;
        }

        transportSource.getNextAudioBlock (bufferToFill);
    }

    void releaseResources() override
    {
        transportSource.releaseResources();
    }

    void resized() override
    {
        openButton          .setBounds (10, 10,  70, 20);
        playButton          .setBounds (10, 40,  70, 20);
        stopButton          .setBounds (10, 70,  70, 20);
        currentPositionLabel.setBounds (8, 95, getWidth() - 20, 20);
        
        Rectangle<int> thumbnailBounds (90, 10, getWidth() - 100, 100);   
        thumbnailComp.setBounds (thumbnailBounds);
        positionOverlay.setBounds (thumbnailBounds);

        processButton       .setBounds (10, 125, 70, 20);
        Rectangle<int> filteredSpectogramBounds (90, 125, getWidth() - 100, 81);   
        filteredSpectrogramComp.setBounds (filteredSpectogramBounds);

        textButton          .setBounds (10, 387, 70, 20);
        infoText            .setBounds (90, 387, getWidth() - 100, 80);

    }

    void changeListenerCallback (ChangeBroadcaster* source) override
    {
        if (source == &transportSource)
        {
            if (transportSource.isPlaying())
                changeState (Playing);
            else if ((state == Stopping) || (state == Playing))
                changeState (Stopped);
            else if (Pausing == state)
                changeState (Paused);
        }
    }

    void timerCallback() override
    {
        RelativeTime position (transportSource.getCurrentPosition());

        auto minutes = ((int) position.inMinutes()) % 60;
        auto seconds = ((int) position.inSeconds()) % 60;
        auto millis  = ((int) position.inMilliseconds()) % 1000;

        auto positionString = String::formatted ("%02d:%02d:%03d", minutes, seconds, millis);

        currentPositionLabel.setText (positionString, dontSendNotification);
    }


private:
    enum TransportState
    {
        Stopped,
        Starting,
        Playing,
        Pausing,
        Paused,
        Stopping
    };

    void changeState (TransportState newState)
    {
        if (state != newState)
        {
            state = newState;

            switch (state)
            {
                case Stopped:
                    playButton.setButtonText ("Play");
                    stopButton.setEnabled (false);
                    playButton.setEnabled (true);
                    transportSource.setPosition (0.0);
                    break;

                case Starting:
                    transportSource.start();
                    break;

                case Playing:
                    playButton.setButtonText ("Pause");
                    stopButton.setEnabled (true);
                    break;

                case Pausing:
                    transportSource.stop();
                    break;
 
                case Paused:
                    playButton.setButtonText ("Resume");
                    break;

                case Stopping:
                    transportSource.stop();
                    currentPositionLabel.setText ("00:00:000", dontSendNotification);
                    break;
            }
        }
    }

    void openButtonClicked()
    {
        FileChooser chooser ("Select a Wave file to play...",
                             {},
                             "*.wav;*.mp3");

        if (chooser.browseForFileToOpen())
        {
            auto file = chooser.getResult();
            auto* reader = formatManager.createReaderFor (file);

            if (reader != nullptr)
            {
                fileBuffer.setSize ((int) reader->numChannels, (int) reader->lengthInSamples); 
                reader->read (&fileBuffer, 0, (int) reader->lengthInSamples, 0, true, true); 
                std::unique_ptr<AudioFormatReaderSource> newSource (new AudioFormatReaderSource (reader, true));
                transportSource.setSource (newSource.get(), 0, nullptr, reader->sampleRate);
                playButton.setEnabled (true);
                processButton.setEnabled(true);
                thumbnailComp.setFile (file);
                readerSource.reset (newSource.release());   
                spectrogramComp.initialize(fileBuffer);    
                filteredSpectrogramComp.initialize(fileBuffer);    

            }
        }
    }

    void playButtonClicked()
    {
        if ((state == Stopped) || (state == Paused))
            changeState (Starting);
        else if (state == Playing)
            changeState (Pausing);
    }

    void stopButtonClicked()
    {
        if (state == Paused)
            changeState (Stopped);
        else
            changeState (Stopping);
    }

    void textButtonClicked()
    {
/*        infoText.setText (vectorToString(spectrogramComp.getValue()), dontSendNotification);
*/    }

    void processButtonClicked()
    {
        spectrogramComp.calculateSTFT();
        filteredSpectrogramComp.filterSpectogram();
        filteredSpectrogramComp.generateFilteredSpectrogramImage();
        infoText.setText (std::to_string(filteredSpectrogramComp.getValue()), dontSendNotification);   

        processButton.setEnabled (false);
        resized();
    }


    //==========================================================================
    TextButton openButton;
    TextButton playButton;
    TextButton stopButton;
    Label currentPositionLabel;
    TextButton textButton;
    TextButton processButton;
    Label infoText;

    AudioFormatManager formatManager;
    AudioSampleBuffer fileBuffer;
    std::unique_ptr<AudioFormatReaderSource> readerSource;
    AudioTransportSource transportSource;
    TransportState state;

    AudioThumbnailCache thumbnailCache;
    SimpleThumbnailComponent thumbnailComp;
    SimplePositionOverlay positionOverlay;

    SpectrogramComponent spectrogramComp;
    FilteredSpectrogramComponent filteredSpectrogramComp;

    torch::jit::script::Module module;


    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainContentComponent)
};
