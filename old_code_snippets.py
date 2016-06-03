from data import process_train_data

# aa = 6  # 9 13 19 28
# size_arr = [9, 13, 19, 28]  # x = int(x * 1.5)
size_arr = [100.0]
for a in size_arr:
    print("hello")
    # size of data used for training

    # x_train, y_train = process_train_data(x_data, y_data, a, 1)

    # # fit the model
    # trace_tmp = mcmc_fit(x_train, y_train)
    #
    # # save the learned mode in pickle file
    # pickle.dump(trace_tmp, open("results/model.pickle", "wb"), protocol=-1)
    #
    # # move plot about learned param values to results folder
    # os.rename("mcmc.png", "results/mcmc_" + data_name_train + "N.png")
