#pragma once

#include <string>

#include "Vertex.h"

bool WriteMesh(Vertex* vertices, unsigned int width, unsigned int height,
               const std::string& filename);
bool WriteMesh(const std::vector<Vertex>& vertices, unsigned int width, unsigned int height, const std::string &filename);
