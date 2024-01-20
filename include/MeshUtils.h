#pragma once

#include "Vertex.h"

bool AreAllEdgesValid(const Vertex* vs[4], float edgeThreshold);
void ProcessVertex(unsigned int idx, unsigned int width, Vertex* vertices,
                   float edgeThreshold, const VertexAction& action);
void ProcessVertex(unsigned int idx, unsigned int width, const std::vector<Vertex>& vertices,
                   float edgeThreshold, const VertexAction &action);