from .utils import *


def power_law_fitting_discrete(time , dx , gridix , observed_data , xlabel , plot , KS_plot):
    #print("Calculating power law fit!")

    def alpha_estimator(observed_data , xmin):
        observed_data = [i for i in observed_data if i >= xmin]

        n = len(observed_data)

        def alpha_estimator_sum(observed_data_xmin , xmin):
            observed_data_xmin = np.array(observed_data_xmin)
            observed_data_xmin = observed_data_xmin / (xmin - 0.5)
            observed_data_xmin = np.log(observed_data_xmin)
            return (1 / np.sum(observed_data_xmin))

        alpha = 1 + (n * alpha_estimator_sum(observed_data , xmin))

        return alpha

    def KS(observed_data , xmin):
        observed_data = [i for i in observed_data if i >= xmin]
        
        dt = np.array(observed_data)
        dt = bincount(dt)
        dt = dt/dt.sum()             #For Normalization
        dt[dt == 0] = np.nan
        dt = pd.DataFrame(dt)
        dt = dt.cumsum(skipna=True)           #To get commulaative distribution
        dt = (1-dt)                    #To get complimentary commulative distribution
        dt = dt[0]          #ccdf_data

        dta = np.array(dt[np.logical_not(np.isnan(dt))])         #Removing all nan values from "dt"----> ccdf of data

        alpha = alpha_estimator(observed_data , xmin)

        observed_data = np.unique(np.sort(np.array(observed_data)))

        ccdf_model = scipy.special.zeta(alpha , observed_data+1)
        ccdf_model = ccdf_model / scipy.special.zeta(alpha , xmin)

        distances_ccdf = np.absolute(dta-ccdf_model)

        KS_statistics = np.amax(distances_ccdf)

        return KS_statistics , ccdf_model


    a = [i for i in observed_data if i >= 1]
    a = np.unique(np.array(a))

    index_for_upper_limit = round(len(a)/1.2)
    xmin_upper_limit = a[index_for_upper_limit]

    #xmin_upper_limit = int(np.percentile(a,95))

    KS_statistics_list = []
    alpha_list = []
    for xmin in range(1,xmin_upper_limit):
        KS_statistics_list.append(KS(observed_data , xmin)[0])
        alpha_list.append(alpha_estimator(observed_data , xmin))

    KS_statistics_list = np.array(KS_statistics_list)
    min_index = np.argmin(KS_statistics_list)

    #print("xmin=" , min_index+1)
    #print("Alpha=" , alpha_list[min_index])

    if(KS_plot == "y"):
        plt.scatter(range(1,xmin_upper_limit) , KS_statistics_list , marker=".")
        plt.xlabel("Xmin")
        plt.ylabel("KS_Statistic")



        KS_sorted_dataframe = pd.DataFrame([KS_statistics_list , alpha_list]).transpose().sort_values(0)
        #print(KS_sorted_dataframe.head(10))


    if(plot == "y"):
        plot_data1 = [i for i in observed_data if i >= min_index+1]

        dt = np.array(plot_data1)
        dt = bincount(dt)
        dt = dt/dt.sum()             #For Normalization
        dt[dt == 0] = np.nan
        dt = pd.DataFrame(dt)
        dt = dt.cumsum(skipna=True)           #To get commulaative distribution
        dt = (1-dt)                    #To get complimentary commulative distribution
        dt = dt[0]          #ccdf_data

        plot_data2 = [i for i in observed_data if i >= min_index+1]
        plot_data2 = np.unique(np.sort(np.array(plot_data2)))

        ax = plt.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim([10**-4 , 10**0])

        plt.plot(plot_data2 , KS(observed_data , min_index+1)[1] , color="black")
        plt.scatter(plot_data2 , dt.iloc[plot_data2] , marker=".")

        plt.xticks(fontsize= 20)
        plt.yticks(fontsize= 20)

        plt.xlabel(xlabel , fontsize=20)
        plt.ylabel("1-CDF" , fontsize=20)

        #plt.text(10**(math.floor(math.log(plot_data2[-1], 10))-1) , 0.5 , f"xmin={str(min_index+1)}" , fontsize=15)
        #plt.text(10**(math.floor(math.log(plot_data2[-1], 10))-1)  , 0.2 , f"alpha={str(round(alpha_list[min_index] , 2))}" , fontsize=15)

        plt.title(f"{str(time)},{str(dx)},{str(gridix)}")

    
    return alpha_list[min_index-1] , min_index , min(KS_statistics_list)







def power_law_fitting_continuous(time , dx , gridix , observed_data , xlabel , plot , KS_plot):
    #print("Calculating power law fit!")

    def alpha_estimator(observed_data , xmin):
        observed_data = [i for i in observed_data if i >= xmin]

        n = len(observed_data)

        def alpha_estimator_sum(observed_data_xmin , xmin):
            observed_data_xmin = np.array(observed_data_xmin)
            observed_data_xmin = observed_data_xmin / (xmin)
            observed_data_xmin = np.log(observed_data_xmin)
            return (1 / np.sum(observed_data_xmin))

        alpha = 1 + (n * alpha_estimator_sum(observed_data , xmin))

        return alpha

    def KS(observed_data , xmin):
        observed_data = [i for i in observed_data if i >= xmin]

        dt = np.array(observed_data)
        dt = np.sort(dt)
        dt = 1 - (np.arange(len(dt)) / float(len(dt)))

        alpha = alpha_estimator(observed_data , xmin)

        #observed_data = np.unique(np.sort(np.array(observed_data)))
        observed_data = np.sort(np.array(observed_data))

        ccdf_model = (observed_data / xmin)**(-alpha+1)

        distances_ccdf = np.absolute(dt-ccdf_model)

        KS_statistics = np.amax(distances_ccdf)

        return KS_statistics , ccdf_model

    xmin_upper_limit = int(round(max(observed_data)-1))

    KS_statistics_list = []
    alpha_list = []
    for xmin in range(1,xmin_upper_limit):
        alpha_list.append(alpha_estimator(observed_data , xmin))
        KS_statistics_list.append(KS(observed_data , xmin)[0])
        
    KS_statistics_list = np.array(KS_statistics_list)
    min_index = np.argmin(KS_statistics_list)

    #print("xmin=" , min_index+1)
    #print("Alpha=" , alpha_list[min_index])

    if(KS_plot == "y"):
        plt.scatter(range(1,xmin_upper_limit) , KS_statistics_list , marker=".")
        plt.xlabel("Xmin")
        plt.ylabel("KS_Statistic")


    if(plot == "y"):
        plot_data1 = [i for i in observed_data if i >= min_index+1]

        dt = np.array(plot_data1)
        dt = np.sort(dt)
        dt = 1 - (np.arange(len(dt)) / float(len(dt)))

        plot_data2 = [i for i in observed_data if i >= min_index+1]
        #plot_data2 = np.unique(np.sort(np.array(plot_data2)))
        plot_data2 = np.sort(np.array(plot_data2))

        ax = plt.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim([10**-4 , 10**0])

        plt.plot(plot_data2 , KS(observed_data , min_index+1)[1] , color="orange")
        plt.scatter(plot_data2[1:] , dt[1:] , marker='.')

        plt.xticks(fontsize= 20)
        plt.yticks(fontsize= 20)

        plt.xlabel(xlabel , fontsize=20)
        plt.ylabel("1-CDF" , fontsize=20)

        #plt.text(10**(math.floor(math.log(plot_data2[-1], 10))-1) , 0.5 , f"xmin={str(min_index+1)}" , fontsize=15)
        #plt.text(10**(math.floor(math.log(plot_data2[-1], 10))-1)  , 0.2 , f"alpha={str(round(alpha_list[min_index] , 2))}" , fontsize=15)

        plt.title(f"{str(time)},{str(dx)},{str(gridix)}")

    return alpha_list[min_index] , min_index+1
