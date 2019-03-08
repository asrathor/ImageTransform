UBIT = '<asrathor>'
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))
from matplotlib import pyplot as plt
# The deep copy was used only to copy values to avoid scenario where both the instances are
# updated even though only one had to. This is because python creates reference while normal copying of
# variables.
from copy import deepcopy
import cv2
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

centers = np.array([[6.2,3.2],[6.6,3.7],[6.5,3.0]])

data_X = np.array([[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]])

# This function is used to plot all the data points to their respective cluster based on the classification vector.
# This was used in both kmeans and bonus task
def plot_data(data,classify_vector,count):

    colors = ['r', 'g', 'b']

    for i in range(data.shape[0]):
        index = classify_vector[i] - 1
        if count == 0:
            plt.scatter(data[i, 0], data[i, 1], s=15, edgecolor=colors[index], marker='^', facecolor='none')
        if count == 1:
            plt.scatter(data[i, 0], data[i, 1], s=15, color=colors[index])
        stri = str('(' + str(data[i, 0]) + ', ' + str(data[i, 1]) + ')')
        if count == 0:
            plt.text(data[i, 0] - 0.06, data[i, 1] - 0.06, stri, fontsize=7)

# This function is used to plot all the centers.
def plot_centers(centers_n):

    colors = ['r', 'g', 'b']
    plt.scatter(centers_n[0, 0], centers_n[0, 1], s=50, c=colors[0])
    stri = str('(' + str(round(centers_n[0, 0], 2)) + ', ' + str(round(centers_n[0, 1], 2)) + ')')
    plt.text(centers_n[0, 0] - 0.06, centers_n[0, 1] - 0.06, stri, fontsize=7)

    plt.scatter(centers_n[1, 0], centers_n[1, 1], s=50, c=colors[1])
    stri = str('(' + str(centers_n[1, 0]) + ', ' + str(centers_n[1, 1]) + ')')
    plt.text(centers_n[1, 0] - 0.06, centers_n[1, 1] - 0.06, stri, fontsize=7)

    plt.scatter(centers_n[2, 0], centers_n[2, 1], s=50, c=colors[2])
    stri = str('(' + str(centers_n[2, 0]) + ', ' + str(centers_n[2, 1]) + ')')
    plt.text(centers_n[2, 0] - 0.06, centers_n[2, 1] - 0.06, stri, fontsize=7)

# This function was used to classify the data points based on the centers.
# The return value is the classification vector.
# Here I used the euclidean distance formula but later I have used np.liang.norm since that is convinient.
def classify(data,centers_n,count):

    dist_arr = []
    classify_vector = []
    for i in range(data.shape[0]):
        euc_dist1 = np.float32(np.sqrt(np.square(data[i][0]-centers_n[0][0])+np.square(data[i][1]-centers_n[0][1])))
        euc_dist2 = np.float32(np.sqrt(np.square(data[i][0]-centers_n[1][0])+np.square(data[i][1]-centers_n[1][1])))
        euc_dist3 = np.float32(np.sqrt(np.square(data[i][0]-centers_n[2][0])+np.square(data[i][1]-centers_n[2][1])))
        print(euc_dist1)

        temp = min(min(euc_dist1,euc_dist2),euc_dist3)
        if temp == euc_dist1:
            dist_arr.append(euc_dist1)
            classify_vector.append(1)
        if temp == euc_dist2:
            dist_arr.append(euc_dist2)
            classify_vector.append(2)
        if temp == euc_dist3:
            dist_arr.append(euc_dist3)
            classify_vector.append(3)

    print(dist_arr)
    print(classify_vector)
    return classify_vector

# This function was used to optimize the centers. First the distance is computed between the centers
# and the data points. Then the position of the minimum distance is taken to assign the new centers.
# Error was computed to see if centers are being properly optimized.
def optimize_center(centers_n):

    new_centers = deepcopy(centers_n)
    error = 0
    dist_arr = np.zeros((10, centers_n.shape[0]))
    count = 0

    while count != 1:
        for i in range(centers.shape[0]):
            dist_arr[:, i] = np.linalg.norm(data_X - centers_n[i], axis=1)
        temp = np.argmin(dist_arr, axis=1)
        previous_centers = deepcopy(new_centers)
        for i in range(3):
            new_centers[i] = np.mean(data_X[temp == i], axis=0)

        error = np.linalg.norm(new_centers - previous_centers)
        count = count + 1

    print(error)
    return new_centers

# This function was used to optimize the color centers. The process is same as optimize_center
# function, however, since the arrays needs to be reshaped with other changes due to different
# shape of the image and centers from previous tasks.
def optimize_center_color(img,centers_n):

    new_centers = deepcopy(centers_n)

    count = 0
    while count != 1:
        euc_dist = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for l in range(centers_n.shape[0]):
                    euc_dist.append(np.linalg.norm(img[i][j][:] - centers_n[l]))
                # smallest_dist_index = np.argmin(euc_dist)
        euc_dist = np.array(euc_dist)
        # print(euc_dist)
        euc_dist = euc_dist.reshape((img.shape[0]*img.shape[1],centers_n.shape[0]))
        # print(euc_dist)
        # print(euc_dist.shape)
        clusters = np.argmin(euc_dist,axis=1)
        old_centers = deepcopy(new_centers)
        clusters = clusters.reshape((img.shape[0],img.shape[1]))
        # print(clusters.shape)
        for i in range(centers_n.shape[0]):
            new_centers[i] = np.mean(img[clusters == i], axis=0)

        error = np.linalg.norm(new_centers - old_centers)
        count = count + 1
        print(error)

    return new_centers

# This function is used to obtain the classification vector for the color centers on the image.
def color_quantization(img,k,col_centers,count):

    classify_vector = np.zeros(img.shape)
    euc_dist = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for l in range(k):
                euc_dist.append(np.linalg.norm(img[i][j][:] - col_centers[l]))
            smallest_dist_index = np.argmin(euc_dist)
            classify_vector[i][j][:] = col_centers[smallest_dist_index]
            euc_dist = []

    return classify_vector

# To run task3.1,3.2 and 3.3.
def run_kmeans():
    # For task 3.1
    classify_vector = classify(data_X,centers,1)
    print('classification vector:',classify_vector)
    plot_data(data_X,classify_vector,0)
    plot_centers(centers)
    plt.savefig('task3_iter1_a.jpg')
    plt.gcf().clear()
    # For task 3.2
    new_centers = optimize_center(centers)
    print('new_centers:',new_centers)
    plot_data(data_X,classify_vector,0)
    plot_centers(new_centers)
    plt.savefig('task3_iter1_b.jpg')
    plt.gcf().clear()
    # For task 3.3(a)
    classify_vector = classify(data_X,new_centers,2)
    print('classification vector:',classify_vector)
    plot_data(data_X,classify_vector,0)
    plot_centers(new_centers)
    plt.savefig('task3_iter2_a.jpg')
    plt.gcf().clear()
    # For task 3.3(b)
    new_centers = optimize_center(new_centers)
    print('new_centers:', new_centers)
    plot_data(data_X,classify_vector,0)
    plot_centers(new_centers)
    plt.savefig('task3_iter2_b.jpg')
    #classify(new_centers,3)

#To run Task3.4. The centers are optimized for 20 times based on TA's post on Piazza.
def run_color_multiple(k):

    img = cv2.imread('baboon.jpg')
    print(img.shape)
    img = img/255
    position_Xarr = []
    position_Yarr = []
    for i in range(k):
        position_Xarr.append(np.random.randint(1, img.shape[0]))
        position_Yarr.append(np.random.randint(1, img.shape[1]))
    col_centers = img[position_Xarr, position_Yarr, :]
    count = 0
    # print(col_centers)

    while count != 20:
        col_centers = optimize_center_color(img,col_centers)
        count = count+1
        print(count)
    classify_vector = color_quantization(img, k, col_centers, count)
    if k == 3:
        classify_vector = classify_vector*255
        cv2.imwrite('task3_baboon_3.jpg', classify_vector)
    if k == 5:
        classify_vector = classify_vector * 255
        cv2.imwrite('task3_baboon_5.jpg', classify_vector)
    if k == 10:
        classify_vector = classify_vector * 255
        cv2.imwrite('task3_baboon_10.jpg', classify_vector)
    if k == 20:
        classify_vector = classify_vector * 255
        cv2.imwrite('task3_baboon_20.jpg', classify_vector)

# This function performs the M-step in GMM.
def expected_maximization(data,p,centers_n):

    new_pi_1 = np.mean(p[:,0])
    new_pi_2 = np.mean(p[:,1])
    new_pi_3 = np.mean(p[:,2])

    new_pi = np.array([new_pi_1,new_pi_2,new_pi_3])

    # ------------------------------------------------------------------

    temp_x = 0.0
    temp_y = 0.0
    for i in range(p.shape[0]):
        temp_x = temp_x + p[i,0]*data[i,0]
        temp_y = temp_y + p[i,0]*data[i,1]

    center_1_x = temp_x/np.sum(p[:,0])
    center_1_y = temp_y/np.sum(p[:,0])

    temp_x = 0.0
    temp_y = 0.0

    for i in range(p.shape[0]):
        temp_x = temp_x + p[i, 1] * data[i, 0]
        temp_y = temp_y + p[i, 1] * data[i, 1]

    center_2_x = temp_x / np.sum(p[:, 1])
    center_2_y = temp_y / np.sum(p[:, 1])

    temp_x = 0.0
    temp_y = 0.0
    for i in range(p.shape[0]):
        temp_x = temp_x + p[i, 2] * data[i, 0]
        temp_y = temp_y + p[i, 2] * data[i, 1]

    center_3_x = temp_x / np.sum(p[:, 2])
    center_3_y = temp_y / np.sum(p[:, 2])

    new_centers = np.array([[center_1_x,center_1_y],[center_2_x,center_2_y],[center_3_x,center_3_y]])

    # -----------------------------------------------------------------

    prod = 0.0

    for i in range(p.shape[0]):

        temp = np.matrix((data[i] - centers_n[0]))
        temp1 = temp.T
        temp2 = temp1.dot(temp)
        prod = prod + p[i,0] * temp2

    new_sigma_1 = prod / np.sum(p[:,0])

    prod = 0.0

    for i in range(p.shape[0]):
        temp = np.matrix((data[i] - centers_n[1]))
        temp1 = temp.T
        temp2 = temp1.dot(temp)
        prod = prod + p[i, 1] * temp2

    new_sigma_2 = prod / np.sum(p[:, 1])

    prod = 0.0

    for i in range(p.shape[0]):
        temp = np.matrix((data[i] - centers_n[2]))
        temp1 = temp.T
        temp2 = temp1.dot(temp)
        prod = prod + p[i, 2] * temp2

    new_sigma_3 = prod / np.sum(p[:, 2])

    new_sigma = np.array([new_sigma_1,new_sigma_2,new_sigma_3])

    return new_pi,new_centers,new_sigma

# This function performs the E-step in GMM.
def gaussian_model(data,sigma,pi,centers_n,count):

    # covariance_matrix = np.cov(data)
    # print('covaraiance: ', covariance_matrix)
    p = np.zeros([data.shape[0],3])
    for k in range(3):
        for i in range(data.shape[0]):
            p[i,k] = pi[k] * multivariate_normal.pdf(data[i],centers_n[k],sigma[k])

    # print(p)

    p_norm = p*np.reciprocal(np.sum(p,1)[None].T)
    if count == 0:
        return p
    if count == 1:
        return p_norm

# To run Task 3.5
def gauss_init(count):

    sigma_1 = np.array([[0.5, 0], [0, 0.5]])
    sigma_2 = np.array([[0.5, 0], [0, 0.5]])
    sigma_3 = np.array([[0.5, 0], [0, 0.5]])

    sigma = np.array([sigma_1, sigma_2, sigma_3])
    pi_1 = 1 / 3
    pi_2 = 1 / 3
    pi_3 = 1 / 3

    pi = np.array([pi_1, pi_2, pi_3])

    p_norm = gaussian_model(data_X,sigma,pi,centers,count)

    new_pi, new_centers, new_sigma = expected_maximization(data_X,p_norm,centers)

    print('Updated centers:', new_centers)
    print('Updated sigma:', new_sigma)
    print('Updated pi:', new_pi)

# Used from the source that is mentioned on the project document.
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

# To classify the old_faithful data
def classify_gmm(p_norm):

    classify_vector = []
    for i in range(p_norm.shape[0]):
        maxim = np.argmax(p_norm[i])
        classify_vector.append(maxim+1)

    return classify_vector

# This function runs task 3.6.
def gauss_oldfaith(count):

    sigma_1 = np.array([[1.30, 13.98], [13.98, 184.82]])
    sigma_2 = np.array([[1.30, 13.98], [13.98, 184.82]])
    sigma_3 = np.array([[1.30, 13.98], [13.98, 184.82]])

    sigma = np.array([sigma_1, sigma_2, sigma_3])

    pi_1 = 1 / 3
    pi_2 = 1 / 3
    pi_3 = 1 / 3

    pi = np.array([pi_1, pi_2, pi_3])

    s = """1       3.600      79
2       1.800      54
3       3.333      74
4       2.283      62
5       4.533      85
6       2.883      55
7       4.700      88
8       3.600      85
9       1.950      51
10      4.350      85
11      1.833      54
12      3.917      84
13      4.200      78
14      1.750      47
15      4.700      83
16      2.167      52
17      1.750      62
18      4.800      84
19      1.600      52
20      4.250      79
21      1.800      51
22      1.750      47
23      3.450      78
24      3.067      69
25      4.533      74
26      3.600      83
27      1.967      55
28      4.083      76
29      3.850      78
30      4.433      79
31      4.300      73
32      4.467      77
33      3.367      66
34      4.033      80
35      3.833      74
36      2.017      52
37      1.867      48
38      4.833      80
39      1.833      59
40      4.783      90
41      4.350      80
42      1.883      58
43      4.567      84
44      1.750      58
45      4.533      73
46      3.317      83
47      3.833      64
48      2.100      53
49      4.633      82
50      2.000      59
51      4.800      75
52      4.716      90
53      1.833      54
54      4.833      80
55      1.733      54
56      4.883      83
57      3.717      71
58      1.667      64
59      4.567      77
60      4.317      81
61      2.233      59
62      4.500      84
63      1.750      48
64      4.800      82
65      1.817      60
66      4.400      92
67      4.167      78
68      4.700      78
69      2.067      65
70      4.700      73
71      4.033      82
72      1.967      56
73      4.500      79
74      4.000      71
75      1.983      62
76      5.067      76
77      2.017      60
78      4.567      78
79      3.883      76
80      3.600      83
81      4.133      75
82      4.333      82
83      4.100      70
84      2.633      65
85      4.067      73
86      4.933      88
87      3.950      76
88      4.517      80
89      2.167      48
90      4.000      86
91      2.200      60
92      4.333      90
93      1.867      50
94      4.817      78
95      1.833      63
96      4.300      72
97      4.667      84
98      3.750      75
99      1.867      51
100     4.900      82
101     2.483      62
102     4.367      88
103     2.100      49
104     4.500      83
105     4.050      81
106     1.867      47
107     4.700      84
108     1.783      52
109     4.850      86
110     3.683      81
111     4.733      75
112     2.300      59
113     4.900      89
114     4.417      79
115     1.700      59
116     4.633      81
117     2.317      50
118     4.600      85
119     1.817      59
120     4.417      87
121     2.617      53
122     4.067      69
123     4.250      77
124     1.967      56
125     4.600      88
126     3.767      81
127     1.917      45
128     4.500      82
129     2.267      55
130     4.650      90
131     1.867      45
132     4.167      83
133     2.800      56
134     4.333      89
135     1.833      46
136     4.383      82
137     1.883      51
138     4.933      86
139     2.033      53
140     3.733      79
141     4.233      81
142     2.233      60
143     4.533      82
144     4.817      77
145     4.333      76
146     1.983      59
147     4.633      80
148     2.017      49
149     5.100      96
150     1.800      53
151     5.033      77
152     4.000      77
153     2.400      65
154     4.600      81
155     3.567      71
156     4.000      70
157     4.500      81
158     4.083      93
159     1.800      53
160     3.967      89
161     2.200      45
162     4.150      86
163     2.000      58
164     3.833      78
165     3.500      66
166     4.583      76
167     2.367      63
168     5.000      88
169     1.933      52
170     4.617      93
171     1.917      49
172     2.083      57
173     4.583      77
174     3.333      68
175     4.167      81
176     4.333      81
177     4.500      73
178     2.417      50
179     4.000      85
180     4.167      74
181     1.883      55
182     4.583      77
183     4.250      83
184     3.767      83
185     2.033      51
186     4.433      78
187     4.083      84
188     1.833      46
189     4.417      83
190     2.183      55
191     4.800      81
192     1.833      57
193     4.800      76
194     4.100      84
195     3.966      77
196     4.233      81
197     3.500      87
198     4.366      77
199     2.250      51
200     4.667      78
201     2.100      60
202     4.350      82
203     4.133      91
204     1.867      53
205     4.600      78
206     1.783      46
207     4.367      77
208     3.850      84
209     1.933      49
210     4.500      83
211     2.383      71
212     4.700      80
213     1.867      49
214     3.833      75
215     3.417      64
216     4.233      76
217     2.400      53
218     4.800      94
219     2.000      55
220     4.150      76
221     1.867      50
222     4.267      82
223     1.750      54
224     4.483      75
225     4.000      78
226     4.117      79
227     4.083      78
228     4.267      78
229     3.917      70
230     4.550      79
231     4.083      70
232     2.417      54
233     4.183      86
234     2.217      50
235     4.450      90
236     1.883      54
237     1.850      54
238     4.283      77
239     3.950      79
240     2.333      64
241     4.150      75
242     2.350      47
243     4.933      86
244     2.900      63
245     4.583      85
246     3.833      82
247     2.083      57
248     4.367      82
249     2.133      67
250     4.350      74
251     2.200      54
252     4.450      83
253     3.567      73
254     4.500      73
255     4.150      88
256     3.817      80
257     3.917      71
258     4.450      83
259     2.000      56
260     4.283      79
261     4.767      78
262     4.533      84
263     1.850      58
264     4.250      83
265     1.983      43
266     2.250      60
267     4.750      75
268     4.117      81
269     2.150      46
270     4.417      90
271     1.817      46
272     4.467      74"""

    X = np.array([float(v) for v in s.split()]).reshape(-1, 3)[:, 1:]
    # print(X)

    centers_n = np.array([[4.0, 81], [2.0, 57], [4.0, 71]])

    p_norm = gaussian_model(X,sigma, pi, centers_n,count)

    print(p_norm.shape)
    print(p_norm)
    # print(p_norm[:,:,0].T)
    # x,y = X.T

    classify_vector = classify_gmm(p_norm)
    plot_data(X,classify_vector,1)

    plot_cov_ellipse(sigma[0],centers_n[0],nstd=2,alpha=0.5,color='red')
    plot_cov_ellipse(sigma[1], centers_n[1], nstd=2, alpha=0.5, color='green')
    plot_cov_ellipse(sigma[2], centers_n[2], nstd=2, alpha=0.5, color='blue')

    plt.savefig('task3_gmm_iter0.jpg')
    plt.gcf().clear()

    new_pi, new_centers, new_sigma = expected_maximization(X, p_norm, centers_n)
    p_norm = gaussian_model(X, new_sigma, new_pi, new_centers,count)

    classify_vector = classify_gmm(p_norm)
    plot_data(X, classify_vector, 1)

    plot_cov_ellipse(new_sigma[0], new_centers[0], nstd=2, alpha=0.5, color='red')
    plot_cov_ellipse(new_sigma[1], new_centers[1], nstd=2, alpha=0.5, color='green')
    plot_cov_ellipse(new_sigma[2], new_centers[2], nstd=2, alpha=0.5, color='blue')

    print('Updated centers:', new_centers)
    print('Updated sigma:', new_sigma)
    print('Updated pi:', new_pi)

    plt.savefig('task3_gmm_iter1.jpg')
    plt.gcf().clear()

    new_pi, new_centers, new_sigma = expected_maximization(X, p_norm, new_centers)
    p_norm = gaussian_model(X, new_sigma, new_pi, new_centers,count)

    classify_vector = classify_gmm(p_norm)
    plot_data(X, classify_vector, 1)

    plot_cov_ellipse(new_sigma[0], new_centers[0], nstd=2, alpha=0.5, color='red')
    plot_cov_ellipse(new_sigma[1], new_centers[1], nstd=2, alpha=0.5, color='green')
    plot_cov_ellipse(new_sigma[2], new_centers[2], nstd=2, alpha=0.5, color='blue')

    print('Updated centers:', new_centers)
    print('Updated sigma:', new_sigma)
    print('Updated pi:', new_pi)

    plt.savefig('task3_gmm_iter2.jpg')
    plt.gcf().clear()

    new_pi, new_centers, new_sigma = expected_maximization(X, p_norm, new_centers)
    p_norm = gaussian_model(X, new_sigma, new_pi, new_centers,count)

    classify_vector = classify_gmm(p_norm)
    plot_data(X, classify_vector, 1)

    plot_cov_ellipse(new_sigma[0], new_centers[0], nstd=2, alpha=0.5, color='red')
    plot_cov_ellipse(new_sigma[1], new_centers[1], nstd=2, alpha=0.5, color='green')
    plot_cov_ellipse(new_sigma[2], new_centers[2], nstd=2, alpha=0.5, color='blue')

    print('Updated centers:', new_centers)
    print('Updated sigma:', new_sigma)
    print('Updated pi:', new_pi)

    plt.savefig('task3_gmm_iter3.jpg')
    plt.gcf().clear()

    new_pi, new_centers, new_sigma = expected_maximization(X, p_norm, new_centers)
    p_norm = gaussian_model(X, new_sigma, new_pi, new_centers,count)

    classify_vector = classify_gmm(p_norm)
    plot_data(X, classify_vector, 1)

    plot_cov_ellipse(new_sigma[0], new_centers[0], nstd=2, alpha=0.5, color='red')
    plot_cov_ellipse(new_sigma[1], new_centers[1], nstd=2, alpha=0.5, color='green')
    plot_cov_ellipse(new_sigma[2], new_centers[2], nstd=2, alpha=0.5, color='blue')

    print('Updated centers:', new_centers)
    print('Updated sigma:', new_sigma)
    print('Updated pi:', new_pi)

    plt.savefig('task3_gmm_iter4.jpg')
    plt.gcf().clear()

    new_pi, new_centers, new_sigma = expected_maximization(X, p_norm, new_centers)
    p_norm = gaussian_model(X, new_sigma, new_pi, new_centers,count)

    classify_vector = classify_gmm(p_norm)
    plot_data(X, classify_vector, 1)

    plot_cov_ellipse(new_sigma[0], new_centers[0], nstd=2, alpha=0.5, color='red')
    plot_cov_ellipse(new_sigma[1], new_centers[1], nstd=2, alpha=0.5, color='green')
    plot_cov_ellipse(new_sigma[2], new_centers[2], nstd=2, alpha=0.5, color='blue')

    print('Updated centers:', new_centers)
    print('Updated sigma:', new_sigma)
    print('Updated pi:', new_pi)

    plt.savefig('task3_gmm_iter5.jpg')
    plt.gcf().clear()

#For task 3.1,3.2,3.3
run_kmeans()
#For task 3.4
run_color_multiple(3)
run_color_multiple(5)
run_color_multiple(10)
run_color_multiple(20)
# For 3.5, using the formula of p(x) given in our lecture slides.
gauss_init(0)
print('--------------------------------------------------------------')
# For 3.5, using the formula of p(x) given in source: http://www.cs.cmu.edu/~aarti/Class/10701_Spring14/slides/EM_annotatedonclass.pdf.
gauss_init(1)
# For task3.6, using the formula of p(x) given in our lecture slides.
#gauss_oldfaith(0)
print('--------------------------------------------------------------')
# For 3.6, using the formula of p(x) given in source: http://www.cs.cmu.edu/~aarti/Class/10701_Spring14/slides/EM_annotatedonclass.pdf.
gauss_oldfaith(1)