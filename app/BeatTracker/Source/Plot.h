#ifndef PLOT_H_INCLUDED
#define PLOT_H_INCLUDED


class GraphPoint
{
public:
    GraphPoint(float xValue, float yValue);

public:
    float xValue;
    float yValue;
    LinkedListPointer<GraphPoint> nextListItem;
};


class GraphDataset
{
public:
    GraphDataset(String label = "Dataset", Colour colour = Colours::black);

    ~GraphDataset();

    void append(GraphPoint* point);

    String label;
    Colour colour;
    LinkedListPointer<GraphPoint>* points;
    LinkedListPointer<GraphDataset> nextListItem;
};


class Graph
{
public:
    Graph(Rectangle<int> region, String title = "Araz Graph", String xLabel = "X-Axis", 
        String yLabel = "Y-Axis", Colour fgColour = Colours::white, Colour bgColour = Colours::black);

    ~Graph();

    void append(GraphDataset* dataset);
    void paint(Graphics& g);

private:
    Rectangle<int> region;
    Rectangle<int> regionGraph;
    String title;
    String xLabel;
    String yLabel;
    Colour fgColour;
    Colour bgColour;
	int xMargin;
	int yMargin;
	LinkedListPointer<GraphDataset>* datasets;
};

#endif // PLOT_H_INCLUDED
