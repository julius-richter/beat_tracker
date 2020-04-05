#include "SimpleThumbnailComponent.h"


SimpleThumbnailComponent::SimpleThumbnailComponent (int sourceSamplesPerThumbnailSample, 
    AudioFormatManager& formatManager, AudioThumbnailCache& cache, 
    BeatIndexComponent& beatIndexComp)
    : thumbnail(sourceSamplesPerThumbnailSample, formatManager, cache)
{
    startTime = 0.0;
    endTime = 120.0;
    timeDifference = 0.0;
    thumbnail.addChangeListener(this);
    beatIndexComp.addChangeListener(this);
    pBeatIndexComp = &beatIndexComp;
}

void SimpleThumbnailComponent::setFile (const File& file)
{
    thumbnail.setSource (new FileInputSource (file));
    startTime = 0.0;
    endTime = thumbnail.getTotalLength();
}

void SimpleThumbnailComponent::paint (Graphics& g)
{
    if (thumbnail.getNumChannels() == 0)
        paintIfNoFileLoaded (g);
    else
        paintIfFileLoaded (g);
}

void SimpleThumbnailComponent::paintIfNoFileLoaded (Graphics& g)
{
    g.fillAll (Colour(0xffe2e1e0));
    // g.setColour (Colours::black);
    // g.drawFittedText ("No File Loaded", getLocalBounds(), Justification::centred, 1);
}

void SimpleThumbnailComponent::paintIfFileLoaded (Graphics& g)
{
    g.fillAll(Colour(0xffe2e1e0));
    g.setColour (Colour(0xff2d3342));

    if (pBeatIndexComp->isMouseButtonDown)
    {        
        positionDifferenceX = pBeatIndexComp->clickPositionDifferenceX;
        positionDifferenceY = pBeatIndexComp->clickPositionDifferenceY;

    
        if (positionDifferenceY > 0.0)
            timeDifference = sgn(positionDifferenceY) * pow(abs(positionDifferenceY / 2.0), 0.5);
        else
            timeDifference = sgn(positionDifferenceY) * pow(abs(positionDifferenceY / 2.0), 1);
    }
    else 
    {
        timeSpan = endTime - timeDifference - (startTime + timeDifference);
        timeShift = timeSpan * (positionDifferenceX / getWidth());

        startTime = std::max(0.0, startTime + timeDifference - timeShift);
        endTime = std::min(endTime - timeDifference - timeShift, thumbnail.getTotalLength());
    
        timeDifference = 0;
        timeSpan = 0;
        positionDifferenceX = 0;

    }

    sendChangeMessage();

    timeSpan = endTime - timeDifference - (startTime + timeDifference);
    timeShift = timeSpan * (positionDifferenceX / getWidth());

    thumbnail.drawChannels (g, getLocalBounds(), std::max(0.0, startTime + timeDifference - timeShift), 
        std::min(endTime - timeDifference - timeShift, thumbnail.getTotalLength()), 1.0f);
}

void SimpleThumbnailComponent::changeListenerCallback (ChangeBroadcaster* source)
{
    if (source == &thumbnail)
        thumbnailChanged();
    if (source == pBeatIndexComp)
        repaint();
}

void SimpleThumbnailComponent::thumbnailChanged()
{
    repaint();
}


