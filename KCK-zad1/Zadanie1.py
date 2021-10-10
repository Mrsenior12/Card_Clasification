import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    names = ["rsel.csv","cel-rs.csv","2cel-rs.csv","cel.csv","2cel.csv"]
    lab = ["1-Evol-RS","1-Coev-RS","2-Coev-RS","1-Coev","2-Coev"]
    marker_shape = ["o","v","D","s","d"]
    wyniki = []
    
# Pierwszy Wykres
    fig,axa = plt.subplots(1,2,figsize=(13,10))
    for i in range(len(names)):
        df = pd.read_csv(names[i])
        axa[0].plot(df["effort"]//1000,(df.iloc[:,2:].mean(axis=1)*100),marker_shape[i], markeredgecolor='k',ls='-',ms=7,markevery=25,label=lab[i]) 
        wyniki.append([i*100.0 for i in df.iloc[-1,2:]])

    axa1 = axa[0].twiny()
    axa1.set_xlim(0,200)
    axa1.set_xticks(range(0,201,40))
    axa1.tick_params(direction='in')

    axa[0].set_xlabel("rozegranych gier (x 1000)")
    axa[0].set_ylabel("Odsetek Wygranych Gier [%]")
    axa1.set_xlabel("Pokolenie")

# Drugi wykres
    axa[1].boxplot(wyniki,notch=True,showfliers=True,showmeans=True,
                    medianprops=dict(linestyle='-',linewidth=1,color='red'),
                    meanprops=dict(marker='o',markersize=7,markeredgecolor='black',markerfacecolor='blue'),
                    whiskerprops=dict(linestyle='-.', color='blue'),
                    boxprops=dict(color='blue'),
                    flierprops=dict(marker='+', markeredgecolor='blue', markerfacecolor='blue'))

# Formatowanie Wykresów    
    axa[0].axis(xmin=0,xmax=500)
    for i in range(2):
        axa[i].set_ylim(60,100)
        axa[i].set_yticks(range(60,101,5))
        axa[i].tick_params(direction='in')
        axa[i].grid(linestyle='--')

    axa[1].set_xticklabels(labels=lab,rotation=30)
    axa[1].yaxis.tick_right()

    axa[0].legend(numpoints=2,loc='lower right',prop={'size':15})
    plt.tight_layout()
#    plt.savefig("zadanie1.jpg")
    plt.show()