# Multi-engine Hybrid Retrosynthesis Report

Intersection size: **2693** steps (aligned across AiZ + 2 synth engines)
Default best chem engine by top-10: **megan**

## Overall top-K accuracy

                                            policy    n   top1   top5  top10  top50
                                   AiZ-USPTO (all) 3028 0.0826 0.1291 0.1522 0.1668
                             AiZ-USPTO (enzymatic) 2258 0.0797 0.1302 0.1590 0.1736
                              AiZ-USPTO (chemical)  770 0.0909 0.1260 0.1325 0.1468
                             EnzExpand-A (979 enz)  979 0.2901 0.4096 0.4229 0.4484
      RootAligned (USPTO-50K, template-free) (all) 3028 0.0779 0.1440 0.1602 0.1991
RootAligned (USPTO-50K, template-free) (enzymatic) 3028 0.0779 0.1440 0.1602 0.1991
 RootAligned (USPTO-50K, template-free) (chemical)    0    NaN    NaN    NaN    NaN
              MEGAN (graph edits, USPTO-50K) (all) 3028 0.0997 0.1456 0.1704 0.2341
        MEGAN (graph edits, USPTO-50K) (enzymatic) 3028 0.0997 0.1456 0.1704 0.2341
         MEGAN (graph edits, USPTO-50K) (chemical)    0    NaN    NaN    NaN    NaN
             UNION chem-engines (3 engs) over 2693 2693 0.1341 0.1831 0.2128 0.2863
                       UNION + EnzExpand over 2693 2693 0.1775 0.2313 0.2536 0.3253
                     ROUTED multi-engine over 2693 2693 0.1289 0.1961 0.2157 0.2818

## By transformation superclass

                  transformation   n  n_enz  aiz_top10  rootaligned_top10  megan_top10  enz_top10  union_all_top10  routed_top10 best_chem_engine
                       oxidation 667    332      0.196              0.216        0.234      0.416            0.301         0.256            megan
                       reduction 427    174      0.429              0.475        0.459      0.672            0.518         0.461      rootaligned
                       amination 263     82      0.023              0.049        0.053      0.073            0.080         0.068            megan
                    C_C_coupling 241     35      0.112              0.100        0.095      0.086            0.137         0.112              aiz
                       acylation 240    102      0.121              0.046        0.050      0.627            0.325         0.292              aiz
                    racemization 222      2      0.009              0.036        0.095      1.000            0.113         0.104            megan
                      hydrolysis 200     75      0.125              0.110        0.175      0.333            0.280         0.165            megan
functional_group_interconversion  84     11      0.036              0.024        0.024      0.091            0.071         0.048              aiz
                 phosphorylation  73     33      0.000              0.000        0.014      0.030            0.027         0.027            megan
                           other  72     20      0.056              0.014        0.014      0.600            0.194         0.181              aiz
                   isomerization  67      6      0.015              0.015        0.060      0.333            0.090         0.090            megan
                   glycosylation  59      2      0.000              0.051        0.000      0.000            0.051         0.051      rootaligned
                  esterification  26      3      0.192              0.346        0.346      1.000            0.385         0.346      rootaligned
              epoxide_hydrolysis  22     14      0.000              0.000        0.000      0.000            0.000         0.000              aiz
                  dehalogenation  20     12      0.200              0.100        0.100      0.333            0.300         0.200              aiz
                       amidation  10      0      0.100              0.100        0.100        NaN            0.100         0.100              aiz