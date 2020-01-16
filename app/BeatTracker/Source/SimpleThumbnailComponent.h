class SimpleThumbnailComponent : public Component,
                                 private ChangeListener
{
public:
    SimpleThumbnailComponent (int sourceSamplesPerThumbnailSample,
                              AudioFormatManager& formatManager,
                              AudioThumbnailCache& cache);

    void setFile (const File& file);

    void paint (Graphics& g) override;

    void paintIfNoFileLoaded (Graphics& g);

    void paintIfFileLoaded (Graphics& g);

    void changeListenerCallback (ChangeBroadcaster* source) override;

private:
    void thumbnailChanged();

    AudioThumbnail thumbnail;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SimpleThumbnailComponent)
};