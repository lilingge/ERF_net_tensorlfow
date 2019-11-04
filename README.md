# ERF_net_tensorlfow
The paper http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf

version:
tensorflow >= 1.4.0
python2.7 or python >= 3.5

Due to confidentiality requirements on the project's code, I hid the pre-training model,
some superparameters have been slightly modified. 
However if you use the default superparameters, that's OK ,you will also get a well result.

if you want to start train this code:
The first you should process cityscapes datasets and imagenet 2012.
You should first run 'sh train_pre_erf_net.sh' to pre-train erf_net model and 
then run 'sh train.sh' to fine tune model on cityscapes dataset.

trainImages.txt, trainLabels.txt, valImages.txt, valLabels.txt like these:

leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
leftImg8bit/train/aachen/aachen_000001_000019_leftImg8bit.png

gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png
gtFine/train/aachen/aachen_000001_000019_gtFine_labelTrainIds.png

leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png
leftImg8bit/val/frankfurt/frankfurt_000000_000576_leftImg8bit.png

gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelTrainIds.png
gtFine/val/frankfurt/frankfurt_000000_000576_gtFine_labelTrainIds.png

And you will run 'sh infer.sh' to inference the model, get visible results

any questions,please touch me : 446049454@qq.com




