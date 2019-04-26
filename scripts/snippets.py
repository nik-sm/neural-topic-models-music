# Get results from the parameter grid search for semi-supervised Scholar
infile = "output/2019_04_25/semi/accuracies.txt"
with open (infile, 'r') as f:
    blob = f.read()

lines = blob.split("params")[1:]
for line in lines:
    #line looks like:  output/2019_04_25/semi/0.2/recon0.0/kl0.0/cl1.0, train accuracy on labels, = 0.1743917732631051
    fields = line.split(",")
    params = fields[0].split("/")
    perc_supervise = params[3]
    reconstr_loss  = params[4].split("recon")[1]
    kl_loss        = params[5].split("kl")[1]
    classify_loss  = params[6].split("cl")[1]
    tr_te = fields[1].split(" ")[1]
    acc = fields[2].split("= ")[1]
    print("train-test: {}, percent_supervise: {}, reconstr_loss: {}, kl_loss: {}, class_loss: {}, acc: {}".format(tr_te, perc_supervise, reconstr_loss, kl_loss, classify_loss, acc))
