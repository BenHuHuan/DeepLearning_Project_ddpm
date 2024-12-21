import pytorch_fid.fid_score
import datetime
import argparse

timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
output_file = f"fid_scores_{timestamp}.txt"

def fid_txt():
    
    with open(output_file, "w") as file:
        for i in range(0,epoch,sample_epoch):
            fid = pytorch_fid.fid_score.calculate_fid_given_paths(
                paths=[path[0],path[1]+'\epoch_{epo}'.format(epo=i)],
                batch_size=batch_size,
                device='cuda',
                dims=2048
            )
            file.write(f"{round(fid, 4)}\n")

parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('--epoch', default=50, type=int, help='epoch')
parser.add_argument('--sample_epoch',  default=5, type=int, help='sample epoch')
parser.add_argument('path',type=str,nargs=2,help='Paths to the generated images')
parser.add_argument('--batch_size',default=20,type=int,help='inceptionv3 batch size')
args = parser.parse_args()

epoch=args.epoch
sample_epoch=args.sample_epoch
path=args.path
batch_size = args.batch_size

if __name__ == '__main__':
    fid_txt()