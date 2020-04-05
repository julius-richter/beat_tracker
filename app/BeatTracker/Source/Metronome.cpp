#include "Metronome.h"


Metronome::Metronome()
{
	formatManager.registerBasicFormats();

	File myFile { File::getSpecialLocation(File::SpecialLocationType::userDesktopDirectory) };
	auto mySpamples = myFile.findChildFiles(File::TypesOfFileToFind::findFiles, true, "click.wav");

	jassert(mySpamples[0].exists());

	auto formatReader = formatManager.createReaderFor(mySpamples[0]);

	metronomeSample.reset( new AudioFormatReaderSource(formatReader, true));

	transportSource.setSource(metronomeSample.get(), 0, nullptr, formatReader->sampleRate);
}

Metronome::~Metronome()
{
	transportSource.~AudioTransportSource();
}


void Metronome::prepareToPlay(int samplesPerBlockExpected, double sampleRate)
{
	if (metronomeSample != nullptr)
	{
		metronomeSample->prepareToPlay(samplesPerBlockExpected, sampleRate);
	}
}


void Metronome::timerCallback()
{

} 


void Metronome::determineBeatIndex()
{
	it = std::lower_bound (beats.begin(), beats.end(), currentPosition); 
	beatIndex = it - beats.begin();
}