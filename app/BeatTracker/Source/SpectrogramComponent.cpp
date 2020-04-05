#include <chrono> 
#include <cmath>
#include <complex>
#include <iostream>
#include <valarray>
#include <vector>

#include "../JuceLibraryCode/JuceHeader.h"
#include "SpectrogramComponent.h"
#include "utils.h"
 

SpectrogramComponent::SpectrogramComponent()
    : forwardFFT (fftOrder),
      spectrogramImage (Image::RGB, 1000, 81, true)
{
    setOpaque (true);
}


SpectrogramComponent::~SpectrogramComponent() {}


void SpectrogramComponent::timerCallback() {}


void SpectrogramComponent::paint(Graphics& g) 
{
    g.fillAll(Colour(0xffe2e1e0));
    g.setOpacity(1.0f);
    g.drawImage(spectrogramImage, getLocalBounds().toFloat());
}


void SpectrogramComponent::initialize(AudioSampleBuffer& fileBuffer)
{
    signal = fileBuffer.getReadPointer(0);
    numSamples = fileBuffer.getNumSamples();
    numFrames = (int) std::floor(((float)numSamples - (float)frameSize) / (float)hopSize + 1.0);
}


void SpectrogramComponent::calculateSTFT()
{
    for (int i = 0; i < numFrames; ++i)
    {
        memcpy(&chunk, signal, sizeof chunk);

        forwardFFT.performFrequencyOnlyForwardTransform (chunk); 


        for (auto &elem : chunk)
        {
            frame.push_back(elem);
        }


        for (auto &elem : chunk) 
            {  
                elem = elem*elem*32767.0*32767.0;
            }
        
        std::vector<float> freqBins(chunk, chunk + frameSize);
        spectogram.push_back(freqBins);
        signal += hopSize;
    }
}


void SpectrogramComponent::filterSpectogram()
{
    std::vector<std::vector<float> > filterbank = createFilterband(numFreqBin);
    std::vector<std::vector<float> > filt(numFrames, std::vector<float>(filterbank.size(), 0.0));
    maxLevel = std::numeric_limits<float>::lowest();

    std::vector<std::vector<int> > filterIndices;
    std::vector<int> indices;

    for (unsigned long i = 0 ; i < filterbank.size() ; ++i)
    {        
        for (int j = 0 ; j < numFreqBin ; ++j)
        {   
            if (filterbank[i][j] > 0.0f)
                indices.push_back(j);
        }
        filterIndices.push_back(indices);
        indices.erase(indices.begin(), indices.end());
    } 

    for (auto t = 0 ; t < numFrames ; ++t)
    {
        for (unsigned long int m = 0 ; m < filterbank.size() ; ++m)
        {   
            filt[t][m] = 0;
            for (unsigned long i = 0 ; i < filterIndices[m].size() ; ++i)
                filt[t][m] += filterbank[m][filterIndices[m][i]] * spectogram[t][filterIndices[m][i]];
            filt[t][m] = log(filt[t][m] + 1);
            maxLevel = maxLevel < filt[t][m] ? filt[t][m] : maxLevel;
        }
    }

    filteredSpectogram = filt;
}


std::vector<std::vector<float> > SpectrogramComponent::createFilterband(int numFreqBin)
{
    const int bandsPerOctavce = 12;
    const float fmin = 30.0; 
    const float fmax = 17000.0;

    std::vector<float> frequencies = logFrequencies(bandsPerOctavce, fmin, fmax);
    std::vector<float> fftFrequencies = calculateFftFrequencies(frameSize, 1.0 / 44100.0);
    std::vector<int> bins = frequenciesToBins(frequencies, fftFrequencies);
    std::vector<std::vector<float> > filterbank = binsToFilterbank(bins, numFreqBin); 
    return filterbank;
}


std::vector<float> SpectrogramComponent::logFrequencies(int bandsPerOctavce, float fmin, float fmax)
{
    const float fref = 440.0f;

    int left = (int) std::floor(log2(fmin / fref) * (float) bandsPerOctavce);
    int right = (int) std::ceil(log2(fmax / fref) * (float) bandsPerOctavce);
    int numBins = right - left;

    std::vector<int> freqRange = range(left, right);
    std::vector<float> frequencies(numBins); 

    for (int i = 0 ; i < numBins; ++i)
    {
        frequencies[i] = fref * pow(2.0f, (float) freqRange[i] / (float) bandsPerOctavce); 
    }

    for (int i = 0 ; i < numBins; ++i)
    {
        if (frequencies[i] < fmin || frequencies[i] > fmax)
            frequencies.erase (frequencies.begin()+i);
    } 
    return frequencies;
} 


std::vector<float> SpectrogramComponent::calculateFftFrequencies(int windowLength, float sampleSpacing)
{
    float val = 1.0 / ((float) windowLength * sampleSpacing);
    float N = (std::floor((windowLength - 1) / 2) + 1);
    std::vector<float> v = range(0.0f, N);
    std::transform(v.begin(), v.end(), v.begin(), [&val](auto& c){return c*val;});
    return v;
}


std::vector<int> SpectrogramComponent::frequenciesToBins(std::vector<float> frequencies, 
    std::vector<float> fftFrequencies)
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


std::vector<std::vector<float> > SpectrogramComponent::binsToFilterbank(std::vector<int> bins, int numFreqBin)
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


void SpectrogramComponent::generateSpectrogramImage()
{
    auto imageHeight = spectrogramImage.getHeight();
    auto imageWidth = spectrogramImage.getWidth();

    for (int x = 0 ; x < imageWidth; ++x)
    {
        for (int y = 0; y < imageHeight; ++y)
        {
            auto fffDataIndexX = (int) ((numFrames - 1) * x / imageWidth);
            float level = filteredSpectogram[fffDataIndexX][y] / maxLevel;
            spectrogramImage.setPixelAt (x, (imageHeight - 1) - y, Colour::fromHSV (level, 1.0f, level, 1.0f));      
        }
    }
}

int SpectrogramComponent::getNumFrames()
{
    return numFrames;
}


float SpectrogramComponent::getValue()
{
    return 0;
}


std::vector<float> SpectrogramComponent::getVector()
{
    /*std::vector<float>::const_iterator first = frame.begin();
    std::vector<float>::const_iterator last = frame.begin();
    std::vector<float> newVec(first+1100, last+1150);
    return newVec;*/
    return spectogram[0];
}


