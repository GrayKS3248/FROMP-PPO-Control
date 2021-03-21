CXX = g++
CXXFLAGS = -fopenmp -Wall -ansi -std=c++2a
INC = -I$(CPYTHONPATH)
LIB_PATH= -L$(PYTHONPATH)
LIB = -lpython39
NAME = FES

auto: 
	$(CXX) $(INC) $(LIB_PATH) $(LIB) -O2 $(CXXFLAGS) Finite_Element_Solver.h Finite_Element_Solver.cpp Train_Autoencoder.cpp -o $(NAME)
	
ppo: 
	$(CXX) $(INC) $(LIB_PATH) $(LIB) -O2 $(CXXFLAGS) Finite_Element_Solver.h Finite_Element_Solver.cpp Train_Controller.cpp -o $(NAME)
	
ppo_debug: 
	$(CXX) $(INC) $(LIB_PATH) $(LIB) -O2 $(CXXFLAGS) -ggdb Finite_Element_Solver.h Finite_Element_Solver.cpp Train_Controller.cpp -o $(NAME)