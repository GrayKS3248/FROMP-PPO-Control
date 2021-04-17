CXX = g++
CXXFLAGS = -fopenmp -Wall -ansi -std=c++2a
INC = -I$(CPYTHONPATH)
LIB_PATH= -L$(PYTHONPATH)
LIB = -lpython39

data: 
	$(CXX) $(INC) $(LIB_PATH) $(LIB) -O2 $(CXXFLAGS) Finite_Element_Solver.h Finite_Element_Solver.cpp Get_AE_Training_Data.cpp -o data

auto: 
	$(CXX) $(INC) $(LIB_PATH) $(LIB) -O2 $(CXXFLAGS) Finite_Element_Solver.h Finite_Element_Solver.cpp Train_Autoencoder.cpp -o auto
	
ppo: 
	$(CXX) $(INC) $(LIB_PATH) $(LIB) -O2 $(CXXFLAGS) Finite_Element_Solver.h Finite_Element_Solver.cpp Train_Controller.cpp -o ppo