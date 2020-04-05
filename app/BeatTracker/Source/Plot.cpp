#include<float.h>

#include "../JuceLibraryCode/JuceHeader.h"
#include "Plot.h"


GraphPoint::GraphPoint(float xValue, float yValue)
{
    this->xValue = xValue;
    this->yValue = yValue;
}


GraphDataset::GraphDataset(String label, Colour colour)
{
    this->label = label;
    this->colour = colour;
    this->points = new LinkedListPointer<GraphPoint>();
}


GraphDataset::~GraphDataset()
{
	points->deleteAll();
    delete points;
}


void GraphDataset::append(GraphPoint* point)
{
    points->append(point);
}


Graph::Graph(Rectangle<int> region, String title, String xLabel, String yLabel, Colour fgColour, Colour bgColour)
	: xMargin(0), yMargin(0)
{
	this->region = region;
	this->regionGraph = region;
	this->regionGraph.expand(xMargin, yMargin);
	this->title = title;
	this->xLabel = xLabel;
	this->yLabel = yLabel;
	this->fgColour = fgColour;
	this->bgColour = bgColour;
	this->datasets = new LinkedListPointer<GraphDataset>();
}


Graph::~Graph()
{
	datasets->deleteAll();
    delete datasets;
}


void Graph::append(GraphDataset* dataset)
{
    datasets->append(dataset);
}


void Graph::paint(Graphics& g)
{
    g.fillAll(bgColour);

	float maxX = FLT_MIN;
	float minX = FLT_MAX;
	// float maxY = FLT_MIN;
	float maxY = 1.0f;
	float minY = 0.0f;
	int lineH = 1;

	GraphDataset* dataset = datasets->get();
	GraphPoint* point = NULL;
	while (dataset != NULL)
	{
		point = dataset->points->get();
		while (point != NULL)
		{
			if (point->xValue > maxX) maxX = point->xValue;
			// if (point->yValue > maxY) maxY = point->yValue;
			if (point->xValue < minX) minX = point->xValue;
			// if (point->yValue < minY) minY = point->yValue;
			point = point->nextListItem;
		}
		dataset = dataset->nextListItem;
	}
	// int dx = (maxX - minX) / 10;
	// int dy = (maxY - minY) / 10;
	float scaleX = regionGraph.getWidth() / ((maxX == minX ? 0.0001 : maxX - minX) * 1.00);
	float scaleY = regionGraph.getHeight() / ((maxY == minY ? 0.0001 : maxY - minY) * 1.00);

	// draw points
	dataset = datasets->get();
    while (dataset != NULL)
    {
		g.setColour(dataset->colour);
		point = dataset->points->get();
		int x = (scaleX * (point->xValue - minX)) + regionGraph.getX();
		int y = regionGraph.getHeight() - (scaleY * (point->yValue - minY)) + regionGraph.getY();
		int preX = x;
		int preY = y;
		while (point != NULL)
		{
			x = (scaleX * (point->xValue - minX)) + regionGraph.getX();
			y = regionGraph.getHeight() - (scaleY * (point->yValue - minY)) + regionGraph.getY();
			g.setColour(Colour(0xff2d3342));
			g.drawLine(preX, preY, x, y, 1);
			// g.fillRect(x - 2, y - 2, 5, 5);
			preX = x;
			preY = y;
			point = point->nextListItem;
		}
		// g.drawSingleLineText(dataset->label, regionGraph.getTopRight().getX() - 5, regionGraph.getTopRight().getY() + (lineH * 18), Justification::right);
		lineH++;
		dataset = dataset->nextListItem;
    }

	// draw x-axis
	// g.setFont(Font(16.0f));
	// g.setColour(fgColour);
	// for (int i = 0; i < 11; i++)
	// {
	// 	int x = (scaleX * i * dx) + regionGraph.getBottomLeft().getX();
	// 	int y = regionGraph.getBottomLeft().getY();
	// 	int y0 = regionGraph.getTopLeft().getY();
	// 	int value = minX + (dx * i);
	// 	Line<float> line(x, y, x, y0);
	// 	float len[] = { 4, 2 };
	// 	g.drawDashedLine(line, len, 2);
	// 	g.drawLine(x, y - 5, x, y + 5, 2);
	// 	g.drawSingleLineText(String(value), x, y + 20, Justification::left);
	// }

	// draw y-axis
	// for (int i = 0; i < 11; i++)
	// {
	// 	int x = regionGraph.getTopLeft().getX();
	// 	int y = regionGraph.getHeight() - (scaleY * i * dy) + regionGraph.getTopLeft().getY();
	// 	int x0 = regionGraph.getTopRight().getX();
	// 	int value = minY + (dy * i);
	// 	Line<float> line(x, y, x0, y);
	// 	float len[] = { 4, 2 };
	// 	g.drawDashedLine(line, len, 2);
	// 	g.drawLine(x - 5, y, x + 5, y, 2);
	// 	g.drawSingleLineText(String(value), x - 5, y, Justification::right);
	// }

    // draw graph general elements
	// g.setColour(fgColour);
	// g.drawRect(regionGraph, 2);
	
	// g.setFont(Font(20.0f));
	// g.setColour(fgColour);
	// g.drawText(title, region, Justification::centredTop, true);
	
	// g.setFont(Font(16.0f));
	// g.setColour(fgColour);
	// g.drawText(xLabel, region, Justification::centredBottom, true);
	//g.addTransform(AffineTransform::identity.rotated(float_Pi / 2.0, region.getCentreX(), region.getCentreY()));
	// g.drawText(yLabel, region, Justification::centredLeft, true);
}
