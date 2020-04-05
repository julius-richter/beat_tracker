#include "../JuceLibraryCode/JuceHeader.h"
#include "LayoutComponent.h"


LayoutComponent::LayoutComponent()
{

}


LayoutComponent::~LayoutComponent()
{

}


void LayoutComponent::paint(Graphics& g) 
{
    g.setColour(Colour(0xff6c6c6c));
    g.fillRoundedRectangle(playerRegion.toFloat(), 5.0f);

    g.setColour(Colour(0xffd4d4d4));
    g.fillRect(currentPositionRegion.toFloat());
	g.setColour(Colour(0xffeaeaea));
    g.drawRect(currentPositionRegion.toFloat(), 1);

}

