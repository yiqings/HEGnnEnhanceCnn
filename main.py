from mm_trainer import MMTrainer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config_path', type=str, default='./config.yaml')
    args = parser.parse_args()       

    mmtrainer = MMTrainer(args.config_path)
    print('start training')
    mmtrainer.train()