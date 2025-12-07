#include "Model.h"
#include "Tensor.h"
#include <numeric>
#include <iomanip>

void PrintTensorShape(Tensor &t)
{
  std::vector<int64_t> shape = t.get_shape();
  for (auto dim : shape)
    printf("DIM %ld\n", dim);
  printf("\n");
}

int main(int argc, char* argv[])
{
  if (argc < 2) {
    printf("Usage %s <saved_model.pb> \n", argv[0]);
    return -1;
  }


  Model model(argv[1]);

  std::vector<std::string> model_ops = model.get_operations();
  for (auto &str : model_ops)
    std::cout << str << std::endl;

  // Frozen model does not have initialization
  //model.init();

  Tensor input{model, "x"};
  Tensor output{model, "Identity"};


  PrintTensorShape(input);
  PrintTensorShape(output);

  std::vector<float> data(200 * 200 * 3);
  input.set_data(data);

  std::cout << "Running!" << std::endl;
  model.run(input, output);
  for (float f : output.get_data<float>()) {
    std::cout << f << " ";
  }
  std::cout << std::endl;
  
  return 0;
}

