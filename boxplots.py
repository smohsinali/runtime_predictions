import numpy as np
import seaborn as sns
import pandas as pd


data = np.loadtxt('boxplot_dt_avg.np')

bx1 = data[:,0]
bx2 = data[:,1]
bx3 = data[:,2]
bx4 = data[:,3]

df = pd.DataFrame({ '9%': bx1,
                    '13%': bx2,
                    '19%': bx3,
                    '28%': bx4})


sns.boxplot(data=df, order=["9%", "13%", "19%", "28%"])
sns.plt.xlabel('Data used')
sns.plt.ylabel('Difference from True value(%)')
sns.plt.ylim(0, 130)
sns.plt.title("Prediction accuracy overview\nDecision Tree: a + bKN(log(N))^2")
sns.plt.show()