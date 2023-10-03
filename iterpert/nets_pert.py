
class Net:
    def __init__(self, params, device, pert_data, fix_evaluation = False):
        self.params = params
        self.device = device
        self.fix_evaluation = fix_evaluation

        from .gears import GEARS
        self.gears_model = GEARS(pert_data, device = device, 
                        weight_bias_track = params['weight_bias_track'], 
                        proj_name = params['wb_proj_name'], 
                        exp_name = params['wb_exp_name'],
                        fix_evaluation = fix_evaluation)

    def train(self, data):
        if not self.params['retrain']:
            raise ValueError('You should turn on the retrain mode')

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
        
    def predict(self, data, detail_eval = False):
        return self.gears_model.predict_from_loader(data, detail_eval)
        
    def get_latent_emb(self, data, latent_type):
        return self.gears_model.get_latent_emb(data, latent_type)
        
    def predict_prob(self, data):
        ## return uncertainty
        if self.params['uncertainty']:
            return self.gears_model.predict_from_loader(data)
        else:
            raise ValueError('Uncertainty is not turned on!')
        
    def get_embeddings(self, data):
        return self.predict(data)
        