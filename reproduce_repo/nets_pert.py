
class Net:
    def __init__(self, params, device, pert_data, model_name, save_dir = None, fix_evaluation = False):
        self.params = params
        self.device = device
        self.model = model_name
        self.fix_evaluation = fix_evaluation

        if model_name == 'GEARS':
            from gears import GEARS
            self.gears_model = GEARS(pert_data, device = device, 
                            weight_bias_track = params['weight_bias_track'], 
                            proj_name = params['wb_proj_name'], 
                            exp_name = params['wb_exp_name'],
                            fix_evaluation = fix_evaluation)

        elif model_name == 'scGPT':
            #import sys
            #sys.path.insert(0, "../scGPT/")
            from scgpt import scGPT_active_interface
            self.scgpt_model = scGPT_active_interface(pert_data, device, save_dir)

    def train(self, data):
        if not self.params['retrain']:
            raise ValueError('You should turn on the retrain mode')
        if self.model == 'GEARS':
            if self.params['simple_loss']:
                print('Use autofocus loss only...')
                uncertainty = False
            else:
                if self.params['uncertainty']:
                    uncertainty = True
                else:
                    uncertainty = False

            self.gears_model.model_initialize(hidden_size = self.params['hidden_size'], 
                                            uncertainty = uncertainty,
                                            uncertainty_reg = self.params['uncertainty_reg'],
                                            direction_lambda = self.params['direction_lambda']
                                            )

            self.gears_model.train(epochs = self.params['epoch_per_cycle'], lr = 1e-3, dataloader = data)
        elif self.model == 'scGPT':
            self.scgpt_model.model_initialize()
            self.scgpt_model.train(epochs = self.params['epoch_per_cycle'], lr = 1e-4, data = data)

    def predict(self, data, detail_eval = False):
        if self.model == 'GEARS':
            return self.gears_model.predict_from_loader(data, detail_eval)
        elif self.model == 'scGPT':
            return self.scgpt_model.predict_from_loader(data, detail_eval)
    
    def get_latent_emb(self, data, latent_type):
        if self.model == 'GEARS':
            return self.gears_model.get_latent_emb(data, latent_type)
        elif self.model == 'scGPT':
            raise ValueError

    def predict_prob(self, data):
        ## return uncertainty
        if self.model == 'GEARS':
            if self.params['uncertainty']:
                return self.gears_model.predict_from_loader(data)
            else:
                raise ValueError('Uncertainty is not turned on!')
        if self.model == 'scGPT':
            raise ValueError('scGPT does not produce uncertainty scores!')
        
    def get_embeddings(self, data):
        return self.predict(data)
        