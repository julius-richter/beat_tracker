#include "ZoomThumbnailComponent.h"


ZoomThumbnailComponent::ZoomThumbnailComponent (int sourceSamplesPerThumbnailSample, 
    AudioFormatManager& formatManager, AudioThumbnailCache& cache, BeatIndexComponent& beatIndexComp,
    SimpleThumbnailComponent& thumbnailComp)
    : thumbnail(sourceSamplesPerThumbnailSample, formatManager, cache)
{
    thumbnail.addChangeListener(this);
    thumbnailComp.addChangeListener(this);
    beatIndexComp.addChangeListener(this);

    pThumbnailComp = &thumbnailComp;
    pBeatIndexComp = &beatIndexComp;
}

void ZoomThumbnailComponent::setFile (const File& file)
{
    thumbnail.setSource (new FileInputSource (file));
}

void ZoomThumbnailComponent::paint (Graphics& g)
{
    if (thumbnail.getNumChannels() == 0)
        paintIfNoFileLoaded (g);
    else
        paintIfFileLoaded (g);
}

void ZoomThumbnailComponent::paintIfNoFileLoaded (Graphics& g)
{  
    int frameThickness = 3;

    g.fillAll(Colour(0xffe2e1e0));

    int y = getLocalBounds().getY();
    width = getLocalBounds().getWidth();
    int height = getLocalBounds().getHeight();

    g.setColour(Colours::black);
    Rectangle<int> frameBounds(0, y, width, height);
    g.drawRect(frameBounds, frameThickness);
}

void ZoomThumbnailComponent::paintIfFileLoaded (Graphics& g)
{
    int frameThickness = 3;

    g.fillAll(Colour(0xffe2e1e0));

    int x = getLocalBounds().getX();
    int y = getLocalBounds().getY();
    width = getLocalBounds().getWidth();
    int height = getLocalBounds().getHeight();

    double startTime = pThumbnailComp->startTime;
    double endTime = pThumbnailComp->endTime;
    double timeSpan = endTime - startTime;

    g.setColour(Colour(0xff2d3342));
    Rectangle<int> thumbnailBounds(x, y + frameThickness, width, height - 2*frameThickness);
    thumbnail.drawChannels (g, thumbnailBounds, 0.0, thumbnail.getTotalLength(), 1.0f);

    frameWidth = (int) (timeSpan / thumbnail.getTotalLength() * (double) width);
    int frameX = (int) (startTime / thumbnail.getTotalLength() * (double) width);

    g.setColour(Colours::black);
    Rectangle<int> frameBounds(frameX, y, frameWidth, height);
    g.drawRect(frameBounds, frameThickness);
}

void ZoomThumbnailComponent::changeListenerCallback(ChangeBroadcaster* source)
{
    if (source == &thumbnail)
        thumbnailChanged();
    if (source == pBeatIndexComp)
        repaint();
}

void ZoomThumbnailComponent::thumbnailChanged()
{
    repaint();
}

void ZoomThumbnailComponent::mouseDown (const MouseEvent& event) 
{   

}

void ZoomThumbnailComponent::mouseDrag (const MouseEvent& event) 
{   

}


