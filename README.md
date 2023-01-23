# Rate-Transient-Analysis-RTA-in-Unconventional-Oil-Gas-Reservoirs-1-
Rate Transient Analysis (RTA) in Unconventional Oil &amp; Gas Reservoirs 




INTRODUCTION


Production from unconventional petroleum reservoirs includes petroleum from 
shale, coal, tight-sand and oil-sand. These reservoirs contain enormous quantities of oil 
and natural gas but pose a technology challenge to both geoscientists and engineers to 
produce economically on a commercial scale. These reservoirs store large volumes and 
are widely distributed at different stratigraphic levels and basin types, offering long-term 
potential for energy supply. Most of these reservoirs are low permeability and porosity 
that need enhancement with hydraulic fracture stimulation to maximize fluid drainage. 
Production from these reservoirs is increasing with continued advancement in geological 
characterization techniques and technology for well drilling, logging, and completion with 
drainage enhancement. Currently, Australia, Argentina, Canada, Egypt, USA, and 
Venezuela are producing natural gas from low permeability reservoirs: tight-sand, shale, 
and coal (CBM). Canada, Russia, USA, and Venezuela are producing heavy oil from oil 
sands. USA is leading the development of techniques for exploring, and technology for 
exploiting unconventional gas resources, which can help to develop potential gas-bearing 
shales of Thailand.
In this project, I will investigate a dataset containing examples of the geological 
features of unconventional reservoirs such as 'Porosity (%)', 'Acoustic impedance 
(kg/m2s*10^6)', 'Brittleness Ratio', 'Vitrinite Reflectance (%)' and understand their 
relationship among each other, their significance and their relationship with the production 
of the reservoir 𝐴√𝐾 which is simply the cross-sectional area multiplied by the square root 
of permeability

DESCRIPTION OF THE DATASET


The Dataset has the following features 'Porosity (%)', 'Acoustic impedance 
(kg/m2s*10^6)', 'Brittleness Ratio', 'Vitrinite Reflectance (%)' and the target feature is 
the 𝐴√𝐾 Aroot(K) as shown in Figure 1

![Screenshot 2023-01-23 155224](https://user-images.githubusercontent.com/118182347/214056614-fee481b0-8e5e-44bc-a260-ced9ec31100f.png)


Figure 1 : Sample of the geological dataset


Porosity is a measure of the void spaces in a material, and is a fraction of the volume 
of voids over the total volume, between 0 and 1, or as a percentage between 0% and 
100%


Matrix Perm “Matrix Permeability” is one of the most important parameters for 
characterizing a source rock reservoir and for predicting hydrocarbon production. The 
low permeability value and the presence of induced fractures during core retrieval and 
transportation make the accurate measurement of the true permeability values for 
source rocks a significant challenge for the industry


Acoustic impedance are measures of the opposition that a system presents to the 
acoustic flow resulting from an acoustic pressure applied to the system.


Brittleness Ratio is the ratio of uniaxial compressive strength to tensile strength.


TOC “Total Organic Carbon” is the amount of carbon found in an organic compound 
and is often used as a non-specific indicator of water quality or cleanliness of 
pharmaceutical manufacturing equipment. TOC may also refer to the amount of organic 
carbon in soil, or in a geological formation, particularly the source rock for a petroleum 
play;


Vitrinite Reflectance is the proportion of incident light reflected from a polished vitrinite 
surface.
𝑨√𝑲 is a productivity metric obtained from rate transient analysis (RTA) in 
unconventional reservoirs and it is equivalent to kh in conventional reservoirs. AOK is 
simply the cross-sectional area multiplied by the square root of permeability
From the dataset we can observe that all features including the target / output feature is 
numeric and that there are no categorical features.


OBJECTIVE


• Understand the dataset and the features
• Feature engineer the features if required
• Predict the production of unconventional reservoirs using different regression 
models


• Evaluate the performance of regression models using 3 cross-validation splits
• Interpret feature importance
• Identify flaws in the model and formulate a plan.


PLAN


Python, seaborn, pandas and numpy will be the tools used to conduct the Exploratory 
Data Analysis, First we will use the .describe() function to understand the dataset more, 
know the mean, standard deviation, minimum values, maximum values, medians for 
each features.

Techniques to apply


• Histogram and density plots to check if there are skewed data
• Boxplots to check features that might have outliers
• Scatter plots will be used to understand the relationship between a feature and 
another
• Heatmap plot to check if there are any features that are linearly correlated

INSIGHTS

From the Histogram and Density plots shown in Figure 2, we can see that all features 
including the target feature are normally distributed, however Matrix Perm (nd) has a 
positive skew, Brittleness ratio and TOC has a negative skew







![histogram](https://user-images.githubusercontent.com/118182347/214057656-c8803f31-e8c7-4fe7-b304-aa327810c169.png)

From the Boxplot as shown below in Figure 3, Outliers exist in the Matrix Permeability 
feature as well as the Vitrinite Reflectance Feature, this can be treated by transforming 
the features

![boxplot](https://user-images.githubusercontent.com/118182347/214058105-dfe9d151-66f8-4720-97f2-a50e86704d20.png)


From the pair plot shown in Figure 4, we can observe that Porosity, Matrix Perm are 
strongly correlated with the target feature Aroot(K), we can also observe that Porosity and 
Matrix Perm are linearly correlated, TOC and Porosity are also linearly correlated, this 
means that we can drop the Matrix Perm and TOC columns as they will not provide 
significant value



![Screenshot 2023-01-23 160054](https://user-images.githubusercontent.com/118182347/214058500-b41f846d-1e44-4fc8-9114-42df207f20e2.png)



