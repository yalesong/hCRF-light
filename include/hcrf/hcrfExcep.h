/*
 hCRF-light Library 2.0 (full version http://hcrf.sf.net)
 Copyright (C) 2012 Yale Song (yalesong@mit.edu)
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef HCRFEXCEP_H
#define HCRFEXCEP_H
#include <stdexcept>

// Thrown when a null pointer is detected in an unexpected place
class HcrfBadPointer : public std::runtime_error
{
public:
  explicit HcrfBadPointer(const std::string& what_arg):
  runtime_error(what_arg) {}
};

// Thrown for feature not implemented
class HcrfNotImplemented : public std::logic_error
{
public:
  explicit HcrfNotImplemented(const std::string& what_arg) :
  logic_error(what_arg) {}
};

class HcrfBadModel : public std::invalid_argument
{
public:
  explicit HcrfBadModel(const std::string& what_arg) :
  invalid_argument(what_arg) {}
};

// Thrown for invalid file name
class BadFileName : public std::invalid_argument
{
public:
  explicit BadFileName(const std::string& what_arg) :
  invalid_argument(what_arg) {}
};

class BadIndex : public std::invalid_argument
{
public:
  explicit BadIndex(const std::string& what_arg) :
  invalid_argument(what_arg) {}
};

class InvalidOptimizer : public std::invalid_argument
{
public:
  explicit InvalidOptimizer(const std::string& what_arg):
  invalid_argument(what_arg) {}
};

class InvalidGradient : public std::invalid_argument
{
public:
  explicit InvalidGradient(const std::string& what_arg):
  invalid_argument(what_arg) {}
};


#endif
