#ifndef METRONOME_H
#define METRONOME_H

#include "../JuceLibraryCode/JuceHeader.h"


class Metronome : public Timer
{
public:
	Metronome();

	~Metronome();

	void prepareToPlay(int samplesPerBlockExpected, double sampleRate);

	void determineBeatIndex();

	void timerCallback() override;

	int beatIndex { 0 };
	double currentPosition;
	std::vector<double> beats;
	AudioTransportSource transportSource;


private:
	AudioFormatManager formatManager;
	std::unique_ptr <AudioFormatReaderSource> metronomeSample { nullptr };
	std::vector<double>::iterator it;

};


#endif 