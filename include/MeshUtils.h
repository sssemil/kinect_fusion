#pragma once

#include "Vertex.h"

bool AreAllEdgesValid(const Vertex* vs[4], float edgeThreshold);
void ProcessVertex(unsigned int idx, unsigned int width, Vertex* vertices,
                   float edgeThreshold, const VertexAction& action);
