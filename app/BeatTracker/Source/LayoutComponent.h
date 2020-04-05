#ifndef LAYOUT_COMPONENT_H
#define LAYOUT_COMPONENT_H


class LayoutComponent : public Component
{
public:
    LayoutComponent();

    ~LayoutComponent() override;

    void paint(Graphics& g) override;

  	Rectangle<int> playerRegion;
  	Rectangle<int> currentPositionRegion;

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LayoutComponent)
};



#endif