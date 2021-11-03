#include "Config_Handler.h"
#include "Speed_Estimator.h"

int main()
{
	Config_Handler* cfg = new Config_Handler("../config_files", "get_ae_data.cfg");
	//cout << cfg->get_orig_cfg() << "\n";
	int test;
	cfg->get_var("actions_per_trajectory", test);
	cout << test << "\n";
	cin.get();
}