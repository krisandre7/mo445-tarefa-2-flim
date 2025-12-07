//
// Created by sergio on 12/05/19.
//

#ifndef CPPFLOW_MODEL_H
#define CPPFLOW_MODEL_H

#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <tuple>
#include <tensorflow/c/c_api.h>
#include "Tensor.h"

class Tensor;

class Model {
public:
    // Pass a path to the model file and optional Tensorflow config options. See examples/load_model/main.cpp.
    explicit Model(  std::string& model_filename,   std::vector<uint8_t>& config_options = {});

    // Rule of five, moving is easy as the pointers can be copied, copying not as i have no idea how to copy
    // the contents of the pointer (i guess dereferencing won't do a deep copy)
    Model(  Model &model) = delete;
    Model(Model &&model) = default;
    Model& operator=(  Model &model) = delete;
    Model& operator=(Model &&model) = default;

    ~Model();

    void init();
    void restore(  std::string& ckpt);
    void save(  std::string& ckpt);
    std::vector<std::string> get_operations()  ;

    // Original Run
    void run(  std::vector<Tensor*>& inputs,   std::vector<Tensor*>& outputs);

    // Run with references
    void run(Tensor& input,   std::vector<Tensor*>& outputs);
    void run(  std::vector<Tensor*>& inputs, Tensor& output);
    void run(Tensor& input, Tensor& output);

    // Run with pointers
    void run(Tensor* input,   std::vector<Tensor*>& outputs);
    void run(  std::vector<Tensor*>& inputs, Tensor* output);
    void run(Tensor* input, Tensor* output);

private:
    TF_Graph* graph;
    TF_Session* session;
    TF_Status* status;

    // Read a file from a string
    static TF_Buffer* read(  std::string&);

    bool status_check(bool throw_exc)  ;
    void error_check(bool condition,   std::string &error)  ;

public:
    friend class Tensor;
};


#endif //CPPFLOW_MODEL_H
