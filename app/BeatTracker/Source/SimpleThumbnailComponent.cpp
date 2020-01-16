#include "../JuceLibraryCode/JuceHeader.h"
#include "SimpleThumbnailComponent.h"



SimpleThumbnailComponent::SimpleThumbnailComponent (int sourceSamplesPerThumbnailSample,
                          AudioFormatManager& formatManager,
                          AudioThumbnailCache& cache)
   : thumbnail (sourceSamplesPerThumbnailSample, formatManager, cache)
{
    thumbnail.addChangeListener (this);
}

void SimpleThumbnailComponent::setFile (const File& file)
{
    thumbnail.setSource (new FileInputSource (file));
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
    g.fillAll (Colours::white);
    g.setColour (Colours::black);
    g.drawFittedText ("No File Loaded", getLocalBounds(), Justification::centred, 1);
}

void SimpleThumbnailComponent::paintIfFileLoaded (Graphics& g)
{
    g.fillAll(Colours::white);

    g.setColour (Colours::blue);
    thumbnail.drawChannels (g, getLocalBounds(), 0.0, thumbnail.getTotalLength(), 1.0f);
}

void SimpleThumbnailComponent::changeListenerCallback (ChangeBroadcaster* source)
{
    if (source == &thumbnail)
        thumbnailChanged();
}

void SimpleThumbnailComponent::thumbnailChanged()
{
    repaint();
}


