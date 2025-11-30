'''
Idea: Using only pictograms to find the most correct match. Quite easy to implement. As a result a matrix should spawn in which the highest number of features matched is hinting into the correct direction.
'''

import cv2 as cv
import matplotlib.pyplot as plt
import time

def ratio_test_bf(param):
    # Apply ratio test: Ratio test only feasable with k=2.
    good = []
    for m,n in param:
        if m.distance < 0.65*n.distance:
            good.append([m])
    return good

def ratio_test_flann(matches):
    #print(matches)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    good = []
    # ratio test as per Lowe's paper
    for i,m_n in enumerate(matches):
        if len(m_n) < 2:  # Skip if less than 2 neighbors
            continue
        m, n = m_n
        if m.distance < 0.65*n.distance:
            matchesMask[i]=[1,0]
            good.append([m])
    return good, matchesMask

def plot_bf(good, pic1, kp1, pic2, kp2):
    #img1 = cv.drawMatchesKnn(inside_cup,akaze_kpi1,inside_robot_desk,akaze_kpi4,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img1 = cv.drawMatchesKnn(pic1,kp1,pic2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img1),plt.show()

def plot_flann(matches, matchesMask, pic1, kp1, pic2, kp2):
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
    img2 = cv.drawMatchesKnn(pic1, kp1, pic2, kp2,matches,None,**draw_params)
    plt.imshow(img2),plt.show()

inside_cup = cv.imread('test_image_inside/inside_cup.jpeg')          # queryImage
inside_plate = cv.imread('test_image_inside/inside_plate.jpeg')          # queryImage
inside_bottle = cv.imread('test_image_inside/inside_bottle.jpeg')          # queryImage
inside_robot_desk_cup = cv.imread('test_image_inside/inside_robot_desk_cup.jpeg') # trainImage
inside_robot_desk_plate = cv.imread('test_image_inside/inside_robot_desk_plate.jpeg') # trainImage
inside_robot_desk_bottle = cv.imread('test_image_inside/inside_robot_desk_bottle.jpeg') # trainImage
inside_robot_chair_cup = cv.imread('test_image_inside/inside_robot_wheelchair_cup.jpeg') # trainImage
inside_robot_chair_plate = cv.imread('test_image_inside/inside_robot_wheelchair_plate.jpeg') # trainImage
inside_robot_chair_bottle = cv.imread('test_image_inside/inside_robot_wheelchair_bottle.jpeg') # trainImage

outside_cup =         cv.imread('test_image_outside/outside_cup.jpeg')          # queryImage
outside_plate =       cv.imread('test_image_outside/outside_plate.jpeg')          # queryImage
outside_bottle =      cv.imread('test_image_outside/outside_bottle.jpeg')          # queryImage
outside_robot_desk_cup =  cv.imread('test_image_outside/outside_robot_desk_cup.jpeg') # trainImage
outside_robot_chair_cup = cv.imread('test_image_outside/outside_robot_wheelchair_cup.jpeg') # trainImage
outside_robot_desk_plate =  cv.imread('test_image_outside/outside_robot_desk_plate.jpeg') # trainImage
outside_robot_chair_plate = cv.imread('test_image_outside/outside_robot_wheelchair_plate.jpeg') # trainImage
outside_robot_desk_bottle =  cv.imread('test_image_outside/outside_robot_desk_bottle.jpeg') # trainImage
outside_robot_chair_bottle = cv.imread('test_image_outside/outside_robot_wheelchair_bottle.jpeg') # trainImage

#### BF-ORB Approach ####
orb = cv.ORB_create()

orb_kpi1, orb_desi1  =  orb.detectAndCompute(inside_cup,None)
orb_kpi2, orb_desi2   = orb.detectAndCompute(inside_plate,None)
orb_kpi3, orb_desi3   = orb.detectAndCompute(inside_bottle,None)
orb_kpi4, orb_desi4 = orb.detectAndCompute(inside_robot_desk_cup,None)
orb_kpi5, orb_desi5 = orb.detectAndCompute(inside_robot_desk_plate,None)
orb_kpi6, orb_desi6 = orb.detectAndCompute(inside_robot_desk_bottle,None)
orb_kpi7, orb_desi7 = orb.detectAndCompute(inside_robot_chair_cup,None)
orb_kpi8, orb_desi8 = orb.detectAndCompute(inside_robot_chair_plate,None)
orb_kpi9, orb_desi9 = orb.detectAndCompute(inside_robot_chair_bottle,None)

orb_kpo1, orb_deso1   = orb.detectAndCompute(outside_cup,None)
orb_kpo2, orb_deso2   = orb.detectAndCompute(outside_plate,None)
orb_kpo3, orb_deso3   = orb.detectAndCompute(outside_bottle,None) 
orb_kpo4, orb_deso4 = orb.detectAndCompute(outside_robot_desk_cup,None)
orb_kpo5, orb_deso5 = orb.detectAndCompute(outside_robot_desk_plate,None)
orb_kpo6, orb_deso6 = orb.detectAndCompute(outside_robot_desk_bottle,None)
orb_kpo7, orb_deso7 = orb.detectAndCompute(outside_robot_chair_cup,None)
orb_kpo8, orb_deso8 = orb.detectAndCompute(outside_robot_chair_plate,None)
orb_kpo9, orb_deso9 = orb.detectAndCompute(outside_robot_chair_bottle,None)

bf = cv.BFMatcher(cv.NORM_HAMMING2)

match1_id = bf.knnMatch(orb_desi1,orb_desi4,k=2)
match2_id = bf.knnMatch(orb_desi1,orb_desi5,k=2)
match3_id = bf.knnMatch(orb_desi1,orb_desi6,k=2)
match4_id = bf.knnMatch(orb_desi2,orb_desi4,k=2)
match5_id = bf.knnMatch(orb_desi2,orb_desi5,k=2)
match6_id = bf.knnMatch(orb_desi2,orb_desi6,k=2)
match7_id = bf.knnMatch(orb_desi3,orb_desi4,k=2)
match8_id = bf.knnMatch(orb_desi3,orb_desi5,k=2)
match9_id = bf.knnMatch(orb_desi3,orb_desi6,k=2)

match1_ic = bf.knnMatch(orb_desi1,orb_desi7,k=2)
match2_ic = bf.knnMatch(orb_desi1,orb_desi8,k=2)
match3_ic = bf.knnMatch(orb_desi1,orb_desi9,k=2)
match4_ic = bf.knnMatch(orb_desi2,orb_desi7,k=2)
match5_ic = bf.knnMatch(orb_desi2,orb_desi8,k=2)
match6_ic = bf.knnMatch(orb_desi2,orb_desi9,k=2)
match7_ic = bf.knnMatch(orb_desi3,orb_desi7,k=2)
match8_ic = bf.knnMatch(orb_desi3,orb_desi8,k=2)
match9_ic = bf.knnMatch(orb_desi3,orb_desi9,k=2)

match1_od = bf.knnMatch(orb_deso1,orb_deso4,k=2)
match2_od = bf.knnMatch(orb_deso1,orb_deso5,k=2)
match3_od = bf.knnMatch(orb_deso1,orb_deso6,k=2)
match4_od = bf.knnMatch(orb_deso2,orb_deso4,k=2)
match5_od = bf.knnMatch(orb_deso2,orb_deso5,k=2)
match6_od = bf.knnMatch(orb_deso2,orb_deso6,k=2)
match7_od = bf.knnMatch(orb_deso3,orb_deso4,k=2)
match8_od = bf.knnMatch(orb_deso3,orb_deso5,k=2)
match9_od = bf.knnMatch(orb_deso3,orb_deso6,k=2)

match1_oc = bf.knnMatch(orb_deso1,orb_deso7,k=2)
match2_oc = bf.knnMatch(orb_deso1,orb_deso8,k=2)
match3_oc = bf.knnMatch(orb_deso1,orb_deso9,k=2)
match4_oc = bf.knnMatch(orb_deso2,orb_deso7,k=2)
match5_oc = bf.knnMatch(orb_deso2,orb_deso8,k=2)
match6_oc = bf.knnMatch(orb_deso2,orb_deso9,k=2)
match7_oc = bf.knnMatch(orb_deso3,orb_deso7,k=2)
match8_oc = bf.knnMatch(orb_deso3,orb_deso8,k=2)
match9_oc = bf.knnMatch(orb_deso3,orb_deso9,k=2)

r_bf_ORB1_id =  ratio_test_bf(match1_id)
r_bf_ORB2_id =  ratio_test_bf(match2_id)
r_bf_ORB3_id =  ratio_test_bf(match3_id)
r_bf_ORB4_id =  ratio_test_bf(match4_id)
r_bf_ORB5_id =  ratio_test_bf(match5_id)
r_bf_ORB6_id =  ratio_test_bf(match6_id)
r_bf_ORB7_id =  ratio_test_bf(match7_id)
r_bf_ORB8_id =  ratio_test_bf(match8_id)
r_bf_ORB9_id =  ratio_test_bf(match9_id)

r_bf_ORB1_ic =  ratio_test_bf(match1_ic)
r_bf_ORB2_ic =  ratio_test_bf(match2_ic)
r_bf_ORB3_ic =  ratio_test_bf(match3_ic)
r_bf_ORB4_ic =  ratio_test_bf(match4_ic)
r_bf_ORB5_ic =  ratio_test_bf(match5_ic)
r_bf_ORB6_ic =  ratio_test_bf(match6_ic)
r_bf_ORB7_ic =  ratio_test_bf(match7_ic)
r_bf_ORB8_ic =  ratio_test_bf(match8_ic)
r_bf_ORB9_ic =  ratio_test_bf(match9_ic)

r_bf_ORB1_od =  ratio_test_bf(match1_od)
r_bf_ORB2_od =  ratio_test_bf(match2_od)
r_bf_ORB3_od =  ratio_test_bf(match3_od)
r_bf_ORB4_od =  ratio_test_bf(match4_od)
r_bf_ORB5_od =  ratio_test_bf(match5_od)
r_bf_ORB6_od =  ratio_test_bf(match6_od)
r_bf_ORB7_od =  ratio_test_bf(match7_od)
r_bf_ORB8_od =  ratio_test_bf(match8_od)
r_bf_ORB9_od =  ratio_test_bf(match9_od)

r_bf_ORB1_oc =  ratio_test_bf(match1_oc)
r_bf_ORB2_oc =  ratio_test_bf(match2_oc)
r_bf_ORB3_oc =  ratio_test_bf(match3_oc)
r_bf_ORB4_oc =  ratio_test_bf(match4_oc)
r_bf_ORB5_oc =  ratio_test_bf(match5_oc)
r_bf_ORB6_oc =  ratio_test_bf(match6_oc)
r_bf_ORB7_oc =  ratio_test_bf(match7_oc)
r_bf_ORB8_oc =  ratio_test_bf(match8_oc)
r_bf_ORB9_oc =  ratio_test_bf(match9_oc)

print(f'Orb inside desk cup permutations: {len(r_bf_ORB1_id)},  {len(r_bf_ORB2_id)}, {len(r_bf_ORB3_id)}') 
print(f'Orb inside desk plate permutations: {len(r_bf_ORB4_id)}, {len(r_bf_ORB5_id)}, {len(r_bf_ORB6_id)}') 
print(f'Orb inside desk bottle permutations: {len(r_bf_ORB7_id)}, {len(r_bf_ORB8_id)}, {len(r_bf_ORB9_id)}')

print(f'Orb inside desk cup permutations: {len(r_bf_ORB1_ic)},  {len(r_bf_ORB2_ic)}, {len(r_bf_ORB3_ic)}') 
print(f'Orb inside desk plate permutations: {len(r_bf_ORB4_ic)}, {len(r_bf_ORB5_ic)}, {len(r_bf_ORB6_ic)}') 
print(f'Orb inside desk bottle permutations: {len(r_bf_ORB7_ic)}, {len(r_bf_ORB8_ic)}, {len(r_bf_ORB9_ic)}')

print(f'Orb inside desk cup permutations: {len(r_bf_ORB1_od)},  {len(r_bf_ORB2_od)}, {len(r_bf_ORB3_od)}') 
print(f'Orb inside desk plate permutations: {len(r_bf_ORB4_od)}, {len(r_bf_ORB5_od)}, {len(r_bf_ORB6_od)}') 
print(f'Orb inside desk bottle permutations: {len(r_bf_ORB7_od)}, {len(r_bf_ORB8_od)}, {len(r_bf_ORB9_od)}')

print(f'Orb inside desk cup permutations: {len(r_bf_ORB1_oc)},  {len(r_bf_ORB2_oc)}, {len(r_bf_ORB3_oc)}') 
print(f'Orb inside desk plate permutations: {len(r_bf_ORB4_oc)}, {len(r_bf_ORB5_oc)}, {len(r_bf_ORB6_oc)}') 
print(f'Orb inside desk bottle permutations: {len(r_bf_ORB7_oc)}, {len(r_bf_ORB8_oc)}, {len(r_bf_ORB9_oc)}')

#### BF-SIFT Approach ####
sift2 = cv.SIFT_create()

sift2_kpi1, sift2_desi1  =  sift2.detectAndCompute(inside_cup,None)
sift2_kpi2, sift2_desi2   = sift2.detectAndCompute(inside_plate,None)
sift2_kpi3, sift2_desi3   = sift2.detectAndCompute(inside_bottle,None)
sift2_kpi4, sift2_desi4 = sift2.detectAndCompute(inside_robot_desk_cup,None)
sift2_kpi5, sift2_desi5 = sift2.detectAndCompute(inside_robot_desk_plate,None)
sift2_kpi6, sift2_desi6 = sift2.detectAndCompute(inside_robot_desk_bottle,None)
sift2_kpi7, sift2_desi7 = sift2.detectAndCompute(inside_robot_chair_cup,None)
sift2_kpi8, sift2_desi8 = sift2.detectAndCompute(inside_robot_chair_plate,None)
sift2_kpi9, sift2_desi9 = sift2.detectAndCompute(inside_robot_chair_bottle,None)

sift2_kpo1, sift2_deso1   = sift2.detectAndCompute(outside_cup,None)
sift2_kpo2, sift2_deso2   = sift2.detectAndCompute(outside_plate,None)
sift2_kpo3, sift2_deso3   = sift2.detectAndCompute(outside_bottle,None) 
sift2_kpo4, sift2_deso4 = sift2.detectAndCompute(outside_robot_desk_cup,None)
sift2_kpo5, sift2_deso5 = sift2.detectAndCompute(outside_robot_desk_plate,None)
sift2_kpo6, sift2_deso6 = sift2.detectAndCompute(outside_robot_desk_bottle,None)
sift2_kpo7, sift2_deso7 = sift2.detectAndCompute(outside_robot_chair_cup,None)
sift2_kpo8, sift2_deso8 = sift2.detectAndCompute(outside_robot_chair_plate,None)
sift2_kpo9, sift2_deso9 = sift2.detectAndCompute(outside_robot_chair_bottle,None)

bf = cv.BFMatcher(cv.NORM_L2)

match1_id = bf.knnMatch(sift2_desi1,sift2_desi4,k=2)
match2_id = bf.knnMatch(sift2_desi1,sift2_desi5,k=2)
match3_id = bf.knnMatch(sift2_desi1,sift2_desi6,k=2)
match4_id = bf.knnMatch(sift2_desi2,sift2_desi4,k=2)
match5_id = bf.knnMatch(sift2_desi2,sift2_desi5,k=2)
match6_id = bf.knnMatch(sift2_desi2,sift2_desi6,k=2)
match7_id = bf.knnMatch(sift2_desi3,sift2_desi4,k=2)
match8_id = bf.knnMatch(sift2_desi3,sift2_desi5,k=2)
match9_id = bf.knnMatch(sift2_desi3,sift2_desi6,k=2)

match1_ic = bf.knnMatch(sift2_desi1,sift2_desi7,k=2)
match2_ic = bf.knnMatch(sift2_desi1,sift2_desi8,k=2)
match3_ic = bf.knnMatch(sift2_desi1,sift2_desi9,k=2)
match4_ic = bf.knnMatch(sift2_desi2,sift2_desi7,k=2)
match5_ic = bf.knnMatch(sift2_desi2,sift2_desi8,k=2)
match6_ic = bf.knnMatch(sift2_desi2,sift2_desi9,k=2)
match7_ic = bf.knnMatch(sift2_desi3,sift2_desi7,k=2)
match8_ic = bf.knnMatch(sift2_desi3,sift2_desi8,k=2)
match9_ic = bf.knnMatch(sift2_desi3,sift2_desi9,k=2)

match1_od = bf.knnMatch(sift2_deso1,sift2_deso4,k=2)
match2_od = bf.knnMatch(sift2_deso1,sift2_deso5,k=2)
match3_od = bf.knnMatch(sift2_deso1,sift2_deso6,k=2)
match4_od = bf.knnMatch(sift2_deso2,sift2_deso4,k=2)
match5_od = bf.knnMatch(sift2_deso2,sift2_deso5,k=2)
match6_od = bf.knnMatch(sift2_deso2,sift2_deso6,k=2)
match7_od = bf.knnMatch(sift2_deso3,sift2_deso4,k=2)
match8_od = bf.knnMatch(sift2_deso3,sift2_deso5,k=2)
match9_od = bf.knnMatch(sift2_deso3,sift2_deso6,k=2)

match1_oc = bf.knnMatch(sift2_deso1,sift2_deso7,k=2)
match2_oc = bf.knnMatch(sift2_deso1,sift2_deso8,k=2)
match3_oc = bf.knnMatch(sift2_deso1,sift2_deso9,k=2)
match4_oc = bf.knnMatch(sift2_deso2,sift2_deso7,k=2)
match5_oc = bf.knnMatch(sift2_deso2,sift2_deso8,k=2)
match6_oc = bf.knnMatch(sift2_deso2,sift2_deso9,k=2)
match7_oc = bf.knnMatch(sift2_deso3,sift2_deso7,k=2)
match8_oc = bf.knnMatch(sift2_deso3,sift2_deso8,k=2)
match9_oc = bf.knnMatch(sift2_deso3,sift2_deso9,k=2)

r_bf_sift21_id =  ratio_test_bf(match1_id)
r_bf_sift22_id =  ratio_test_bf(match2_id)
r_bf_sift23_id =  ratio_test_bf(match3_id)
r_bf_sift24_id =  ratio_test_bf(match4_id)
r_bf_sift25_id =  ratio_test_bf(match5_id)
r_bf_sift26_id =  ratio_test_bf(match6_id)
r_bf_sift27_id =  ratio_test_bf(match7_id)
r_bf_sift28_id =  ratio_test_bf(match8_id)
r_bf_sift29_id =  ratio_test_bf(match9_id)

r_bf_sift21_ic =  ratio_test_bf(match1_ic)
r_bf_sift22_ic =  ratio_test_bf(match2_ic)
r_bf_sift23_ic =  ratio_test_bf(match3_ic)
r_bf_sift24_ic =  ratio_test_bf(match4_ic)
r_bf_sift25_ic =  ratio_test_bf(match5_ic)
r_bf_sift26_ic =  ratio_test_bf(match6_ic)
r_bf_sift27_ic =  ratio_test_bf(match7_ic)
r_bf_sift28_ic =  ratio_test_bf(match8_ic)
r_bf_sift29_ic =  ratio_test_bf(match9_ic)

r_bf_sift21_od =  ratio_test_bf(match1_od)
r_bf_sift22_od =  ratio_test_bf(match2_od)
r_bf_sift23_od =  ratio_test_bf(match3_od)
r_bf_sift24_od =  ratio_test_bf(match4_od)
r_bf_sift25_od =  ratio_test_bf(match5_od)
r_bf_sift26_od =  ratio_test_bf(match6_od)
r_bf_sift27_od =  ratio_test_bf(match7_od)
r_bf_sift28_od =  ratio_test_bf(match8_od)
r_bf_sift29_od =  ratio_test_bf(match9_od)

r_bf_sift21_oc =  ratio_test_bf(match1_oc)
r_bf_sift22_oc =  ratio_test_bf(match2_oc)
r_bf_sift23_oc =  ratio_test_bf(match3_oc)
r_bf_sift24_oc =  ratio_test_bf(match4_oc)
r_bf_sift25_oc =  ratio_test_bf(match5_oc)
r_bf_sift26_oc =  ratio_test_bf(match6_oc)
r_bf_sift27_oc =  ratio_test_bf(match7_oc)
r_bf_sift28_oc =  ratio_test_bf(match8_oc)
r_bf_sift29_oc =  ratio_test_bf(match9_oc)

print(f'sift2 inside desk cup permutations: {len(r_bf_sift21_id)},  {len(r_bf_sift22_id)}, {len(r_bf_sift23_id)}') 
print(f'sift2 inside desk plate permutations: {len(r_bf_sift24_id)}, {len(r_bf_sift25_id)}, {len(r_bf_sift26_id)}') 
print(f'sift2 inside desk bottle permutations: {len(r_bf_sift27_id)}, {len(r_bf_sift28_id)}, {len(r_bf_sift29_id)}')

print(f'sift2 inside desk cup permutations: {len(r_bf_sift21_ic)},  {len(r_bf_sift22_ic)}, {len(r_bf_sift23_ic)}') 
print(f'sift2 inside desk plate permutations: {len(r_bf_sift24_ic)}, {len(r_bf_sift25_ic)}, {len(r_bf_sift26_ic)}') 
print(f'sift2 inside desk bottle permutations: {len(r_bf_sift27_ic)}, {len(r_bf_sift28_ic)}, {len(r_bf_sift29_ic)}')

print(f'sift2 inside desk cup permutations: {len(r_bf_sift21_od)},  {len(r_bf_sift22_od)}, {len(r_bf_sift23_od)}') 
print(f'sift2 inside desk plate permutations: {len(r_bf_sift24_od)}, {len(r_bf_sift25_od)}, {len(r_bf_sift26_od)}') 
print(f'sift2 inside desk bottle permutations: {len(r_bf_sift27_od)}, {len(r_bf_sift28_od)}, {len(r_bf_sift29_od)}')

print(f'sift2 inside desk cup permutations: {len(r_bf_sift21_oc)},  {len(r_bf_sift22_oc)}, {len(r_bf_sift23_oc)}') 
print(f'sift2 inside desk plate permutations: {len(r_bf_sift24_oc)}, {len(r_bf_sift25_oc)}, {len(r_bf_sift26_oc)}') 
print(f'sift2 inside desk bottle permutations: {len(r_bf_sift27_oc)}, {len(r_bf_sift28_oc)}, {len(r_bf_sift29_oc)}')

#### AKAZE approach #####
# Feature detection
akaze = cv.AKAZE_create()

akaze_desi = [0,0,0]
akaze_desi_d = [0,0,0]
akaze_desi_c = [0,0,0]

akaze_deso = [0,0,0]
akaze_deso_d = [0,0,0]
akaze_deso_c = [0,0,0]

akaze_kpi1, akaze_desi[0]   = akaze.detectAndCompute(inside_cup,None)
akaze_kpi2, akaze_desi[1]   = akaze.detectAndCompute(inside_plate,None)
akaze_kpi3, akaze_desi[2]   = akaze.detectAndCompute(inside_bottle,None)
akaze_kpi4, akaze_desi_d[0] = akaze.detectAndCompute(inside_robot_desk_cup,None)
akaze_kpi5, akaze_desi_d[1] = akaze.detectAndCompute(inside_robot_desk_plate,None)
akaze_kpi6, akaze_desi_d[2] = akaze.detectAndCompute(inside_robot_desk_bottle,None)
akaze_kpi7, akaze_desi_c[0] = akaze.detectAndCompute(inside_robot_chair_cup,None)
akaze_kpi8, akaze_desi_c[1] = akaze.detectAndCompute(inside_robot_chair_plate,None)
akaze_kpi9, akaze_desi_c[2] = akaze.detectAndCompute(inside_robot_chair_bottle,None)

akaze_kpo1, akaze_deso[0]   = akaze.detectAndCompute(outside_cup,None)
akaze_kpo2, akaze_deso[1]   = akaze.detectAndCompute(outside_plate,None)
akaze_kpo3, akaze_deso[2]   = akaze.detectAndCompute(outside_bottle,None) 
akaze_kpo4, akaze_deso_d[0] = akaze.detectAndCompute(outside_robot_desk_cup,None)
akaze_kpo5, akaze_deso_d[1] = akaze.detectAndCompute(outside_robot_desk_plate,None)
akaze_kpo6, akaze_deso_d[2] = akaze.detectAndCompute(outside_robot_desk_bottle,None)
akaze_kpo7, akaze_deso_c[0] = akaze.detectAndCompute(outside_robot_chair_cup,None)
akaze_kpo8, akaze_deso_c[1] = akaze.detectAndCompute(outside_robot_chair_plate,None)
akaze_kpo9, akaze_deso_c[2] = akaze.detectAndCompute(outside_robot_chair_bottle,None)

result1 = cv.drawKeypoints(inside_cup, akaze_kpi1, None)
cv.imwrite("features1.jpg", result1)


# feature matching
FLANN_INDEX_LSH = 6
index_params = dict(algorithm = FLANN_INDEX_LSH,  
                    table_number=6,      # number of hash tables (e.g., 6–12)
                    key_size=12,         # hash key size in bits (e.g., 12–20)
                    multi_probe_level=1) # how many nearby buckets to check (higher=more recall)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)

akaze_permutations_inside_desk = []
akaze_permutations_inside_chair = []
akaze_permutations_outside_desk = []
akaze_permutations_outside_chair = []

for i in akaze_desi:
    for n in akaze_desi_d:
        akaze_permutations_inside_desk.append(flann.knnMatch(i,n,k=2))
    for m in akaze_desi_c:
        akaze_permutations_inside_chair.append(flann.knnMatch(i,m,k=2))

for i in akaze_deso:
    for n in akaze_deso_d:
        akaze_permutations_outside_desk.append(flann.knnMatch(i,n,k=2))
    for m in akaze_deso_c:
        akaze_permutations_outside_chair.append(flann.knnMatch(i,m,k=2))

#print(f'Test: {akaze_permutations_inside_desk[0]}')
print(f'AKAZE Permutations Inside Desk: {len(akaze_permutations_inside_desk[0])}, {len(akaze_permutations_inside_desk[1])}, {len(akaze_permutations_inside_desk[2])}, {len(akaze_permutations_inside_desk[3])}, {len(akaze_permutations_inside_desk[4])}, {len(akaze_permutations_inside_desk[5])}, {len(akaze_permutations_inside_desk[6])}, {len(akaze_permutations_inside_desk[7])}, {len(akaze_permutations_inside_desk[8])}')
print(f'AKAZE Permutations Inside Chair: {len(akaze_permutations_inside_chair[0])}, {len(akaze_permutations_inside_chair[1])}, {len(akaze_permutations_inside_chair[2])},{len(akaze_permutations_inside_chair[3])}, {len(akaze_permutations_inside_chair[4])}, {len(akaze_permutations_inside_chair[5])}, {len(akaze_permutations_inside_chair[6])}, {len(akaze_permutations_inside_chair[7])}, {len(akaze_permutations_inside_chair[8])}')

print(f'AKAZE Permutations outside Desk: {len(akaze_permutations_outside_desk[0])}, {len(akaze_permutations_outside_desk[1])}, {len(akaze_permutations_outside_desk[2])}, {len(akaze_permutations_outside_desk[3])}, {len(akaze_permutations_outside_desk[4])}, {len(akaze_permutations_outside_desk[5])}, {len(akaze_permutations_outside_desk[6])}, {len(akaze_permutations_outside_desk[7])}, {len(akaze_permutations_outside_desk[8])}')
print(f'AKAZE Permutations outside Chair: {len(akaze_permutations_outside_chair[0])}, {len(akaze_permutations_outside_chair[1])}, {len(akaze_permutations_outside_chair[2])},{len(akaze_permutations_outside_chair[3])}, {len(akaze_permutations_outside_chair[4])}, {len(akaze_permutations_outside_chair[5])}, {len(akaze_permutations_outside_chair[6])}, {len(akaze_permutations_outside_chair[7])}, {len(akaze_permutations_outside_chair[8])}')

r_id_akaze = [0,0,0,0,0,0,0,0,0]
matches_mask_id_akaze = [0,0,0,0,0,0,0,0,0]
for x in range(len(r_id_akaze)):
       r_id_akaze[x], matches_mask_id_akaze[x] =  ratio_test_flann(akaze_permutations_inside_desk[x])

print(f'AKAZE inside desk cup permutations: {len(r_id_akaze[0])},  {len(r_id_akaze[1])}, {len(r_id_akaze[2])}') 
print(f'AKAZE inside desk plate permutations: {len(r_id_akaze[3])}, {len(r_id_akaze[4])}, {len(r_id_akaze[5])}') 
print(f'AKAZE inside desk bottle permutations: {len(r_id_akaze[6])}, {len(r_id_akaze[7])}, {len(r_id_akaze[8])}')

r_ic_akaze = [0,0,0,0,0,0,0,0,0]
matches_mask_ic_akaze = [0,0,0,0,0,0,0,0,0]
for x in range(len(r_ic_akaze)):
       r_ic_akaze[x], matches_mask_ic_akaze[x] =  ratio_test_flann(akaze_permutations_inside_chair[x])

print(f'AKAZE inside chair cup permutations: {len(r_ic_akaze[0])},  {len(r_ic_akaze[1])}, {len(r_ic_akaze[2])}') 
print(f'AKAZE insice chair plate permutations: {len(r_ic_akaze[3])}, {len(r_ic_akaze[4])}, {len(r_ic_akaze[5])}') 
print(f'AKAZE insice chair bottle permutations: {len(r_ic_akaze[6])}, {len(r_ic_akaze[7])}, {len(r_ic_akaze[8])}')

r_od_akaze = [0,0,0,0,0,0,0,0,0]
matches_mask_od_akaze = [0,0,0,0,0,0,0,0,0]
for x in range(len(r_od_akaze)):
       r_od_akaze[x], matches_mask_od_akaze[x] =  ratio_test_flann(akaze_permutations_outside_desk[x])


print(f'AKAZE outside desk cup permutations: {len(r_od_akaze[0])},  {len(r_od_akaze[1])}, {len(r_od_akaze[2])}') 
print(f'AKAZE outside desk plate permutations: {len(r_od_akaze[3])}, {len(r_od_akaze[4])}, {len(r_od_akaze[5])}') 
print(f'AKAZE outside desk bottle permutations: {len(r_od_akaze[6])}, {len(r_od_akaze[7])}, {len(r_od_akaze[8])}')

r_oc_akaze = [0,0,0,0,0,0,0,0,0]
matches_mask_oc_akaze = [0,0,0,0,0,0,0,0,0]
for x in range(len(r_oc_akaze)):
       r_oc_akaze[x], matches_mask_oc_akaze[x] =  ratio_test_flann(akaze_permutations_outside_chair[x])

print(f'AKAZE outside chair cup permutations: {len(r_oc_akaze[0])},  {len(r_oc_akaze[1])}, {len(r_oc_akaze[2])}') 
print(f'AKAZE outside chair plate permutations: {len(r_oc_akaze[3])}, {len(r_oc_akaze[4])}, {len(r_oc_akaze[5])}') 
print(f'AKAZE outside chair bottle permutations: {len(r_oc_akaze[6])}, {len(r_oc_akaze[7])}, {len(r_oc_akaze[8])}')

'''
#Working plots
plot_flann(akaze_permutations_inside_desk[0], matches_mask_id_akaze[0], inside_cup, akaze_kpi1, inside_robot_desk_cup, akaze_kpi4)
plot_flann(akaze_permutations_inside_desk[1], matches_mask_id_akaze[1], inside_cup, akaze_kpi1, inside_robot_desk_plate, akaze_kpi5)
plot_flann(akaze_permutations_inside_desk[2], matches_mask_id_akaze[2], inside_cup, akaze_kpi1, inside_robot_desk_bottle, akaze_kpi6)
plot_flann(akaze_permutations_inside_desk[3], matches_mask_id_akaze[3], inside_plate, akaze_kpi2, inside_robot_desk_cup, akaze_kpi4)
plot_flann(akaze_permutations_inside_desk[4], matches_mask_id_akaze[4], inside_plate, akaze_kpi2, inside_robot_desk_plate, akaze_kpi5)
plot_flann(akaze_permutations_inside_desk[5], matches_mask_id_akaze[5], inside_plate, akaze_kpi2, inside_robot_desk_bottle, akaze_kpi6)
plot_flann(akaze_permutations_inside_desk[6], matches_mask_id_akaze[6], inside_bottle, akaze_kpi3, inside_robot_desk_cup, akaze_kpi4)
plot_flann(akaze_permutations_inside_desk[7], matches_mask_id_akaze[7], inside_bottle, akaze_kpi3, inside_robot_desk_plate, akaze_kpi5)
plot_flann(akaze_permutations_inside_desk[8], matches_mask_id_akaze[8], inside_bottle, akaze_kpi3, inside_robot_desk_bottle, akaze_kpi6)
'''

#### SIFT-FLANN ####
#Feature detection

sift = cv.SIFT_create()

sift_desi = [0,0,0]
sift_desi_d = [0,0,0]
sift_desi_c = [0,0,0]

sift_deso = [0,0,0]
sift_deso_d = [0,0,0]
sift_deso_c = [0,0,0]

sift_kpi1, sift_desi[0] = sift.detectAndCompute(inside_cup,None)
sift_kpi2, sift_desi[1] = sift.detectAndCompute(inside_plate,None)
sift_kpi3, sift_desi[2] = sift.detectAndCompute(inside_bottle,None)
sift_kpi4, sift_desi_d[0] = sift.detectAndCompute(inside_robot_desk_cup,None)
sift_kpi5, sift_desi_d[1] = sift.detectAndCompute(inside_robot_desk_plate,None)
sift_kpi6, sift_desi_d[2] = sift.detectAndCompute(inside_robot_desk_bottle,None)
sift_kpi7, sift_desi_c[0] = sift.detectAndCompute(inside_robot_chair_cup,None)
sift_kpi8, sift_desi_c[1] = sift.detectAndCompute(inside_robot_chair_plate,None)
sift_kpi9, sift_desi_c[2] = sift.detectAndCompute(inside_robot_chair_bottle,None)

sift_kpo1, sift_deso[0] = sift.detectAndCompute(outside_cup,None)
sift_kpo2, sift_deso[1] = sift.detectAndCompute(outside_plate,None)
sift_kpo3, sift_deso[2] = sift.detectAndCompute(outside_bottle,None) 
sift_kpo4, sift_deso_d[0] = sift.detectAndCompute(outside_robot_desk_cup,None)
sift_kpo5, sift_deso_d[1] = sift.detectAndCompute(outside_robot_desk_plate,None)
sift_kpo6, sift_deso_d[2] = sift.detectAndCompute(outside_robot_desk_bottle,None)
sift_kpo7, sift_deso_c[0] = sift.detectAndCompute(outside_robot_chair_cup,None)
sift_kpo8, sift_deso_c[1] = sift.detectAndCompute(outside_robot_chair_plate,None)
sift_kpo9, sift_deso_c[2] = sift.detectAndCompute(outside_robot_chair_bottle,None)

# Feature Matching

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)

sift_permutations_inside_desk = []
sift_permutations_inside_chair = []
sift_permutations_outside_desk = []
sift_permutations_outside_chair = []

for i in sift_desi:
    for n in sift_desi_d:
        sift_permutations_inside_desk.append(flann.knnMatch(i,n,k=2))
    for m in sift_desi_c:
        sift_permutations_inside_chair.append(flann.knnMatch(i,m,k=2))

for i in sift_deso:
    for n in sift_deso_d:
        sift_permutations_outside_desk.append(flann.knnMatch(i,n,k=2))
    for m in sift_deso_c:
        sift_permutations_outside_chair.append(flann.knnMatch(i,m,k=2))

print(f'sift Permutations Inside Desk: {len(sift_permutations_inside_desk[0])}, {len(sift_permutations_inside_desk[1])}, {len(sift_permutations_inside_desk[2])}, {len(sift_permutations_inside_desk[3])}, {len(sift_permutations_inside_desk[4])}, {len(sift_permutations_inside_desk[5])}, {len(sift_permutations_inside_desk[6])}, {len(sift_permutations_inside_desk[7])}, {len(sift_permutations_inside_desk[8])}')
print(f'sift Permutations Inside Chair: {len(sift_permutations_inside_chair[0])}, {len(sift_permutations_inside_chair[1])}, {len(sift_permutations_inside_chair[2])},{len(sift_permutations_inside_chair[3])}, {len(sift_permutations_inside_chair[4])}, {len(sift_permutations_inside_chair[5])}, {len(sift_permutations_inside_chair[6])}, {len(sift_permutations_inside_chair[7])}, {len(sift_permutations_inside_chair[8])}')

print(f'sift Permutations outside Desk: {len(sift_permutations_outside_desk[0])}, {len(sift_permutations_outside_desk[1])}, {len(sift_permutations_outside_desk[2])}, {len(sift_permutations_outside_desk[3])}, {len(sift_permutations_outside_desk[4])}, {len(sift_permutations_outside_desk[5])}, {len(sift_permutations_outside_desk[6])}, {len(sift_permutations_outside_desk[7])}, {len(sift_permutations_outside_desk[8])}')
print(f'sift Permutations outside Chair: {len(sift_permutations_outside_chair[0])}, {len(sift_permutations_outside_chair[1])}, {len(sift_permutations_outside_chair[2])},{len(sift_permutations_outside_chair[3])}, {len(sift_permutations_outside_chair[4])}, {len(sift_permutations_outside_chair[5])}, {len(sift_permutations_outside_chair[6])}, {len(sift_permutations_outside_chair[7])}, {len(sift_permutations_outside_chair[8])}')


r_id_sift = [0,0,0,0,0,0,0,0,0]
matches_mask_id_sift = [0,0,0,0,0,0,0,0,0]
for x in range(len(r_id_sift)):
       r_id_sift[x], matches_mask_id_sift[x] =  ratio_test_flann(sift_permutations_inside_desk[x])

print(f'sift inside desk cup permutations: {len(r_id_sift[0])},  {len(r_id_sift[1])}, {len(r_id_sift[2])}') 
print(f'sift inside desk plate permutations: {len(r_id_sift[3])}, {len(r_id_sift[4])}, {len(r_id_sift[5])}') 
print(f'sift inside desk bottle permutations: {len(r_id_sift[6])}, {len(r_id_sift[7])}, {len(r_id_sift[8])}')

r_ic_sift = [0,0,0,0,0,0,0,0,0]
matches_mask_ic_sift = [0,0,0,0,0,0,0,0,0]
for x in range(len(r_ic_sift)):
       r_ic_sift[x], matches_mask_ic_sift[x] =  ratio_test_flann(sift_permutations_inside_chair[x])

print(f'sift inside chair cup permutations: {len(r_ic_sift[0])},  {len(r_ic_sift[1])}, {len(r_ic_sift[2])}') 
print(f'sift insice chair plate permutations: {len(r_ic_sift[3])}, {len(r_ic_sift[4])}, {len(r_ic_sift[5])}') 
print(f'sift insice chair bottle permutations: {len(r_ic_sift[6])}, {len(r_ic_sift[7])}, {len(r_ic_sift[8])}')

r_od_sift = [0,0,0,0,0,0,0,0,0]
matches_mask_od_sift = [0,0,0,0,0,0,0,0,0]
for x in range(len(r_od_sift)):
       r_od_sift[x], matches_mask_od_sift[x] =  ratio_test_flann(sift_permutations_outside_desk[x])


print(f'sift outside desk cup permutations: {len(r_od_sift[0])},  {len(r_od_sift[1])}, {len(r_od_sift[2])}') 
print(f'sift outside desk plate permutations: {len(r_od_sift[3])}, {len(r_od_sift[4])}, {len(r_od_sift[5])}') 
print(f'sift outside desk bottle permutations: {len(r_od_sift[6])}, {len(r_od_sift[7])}, {len(r_od_sift[8])}')

r_oc_sift = [0,0,0,0,0,0,0,0,0]
matches_mask_oc_sift = [0,0,0,0,0,0,0,0,0]
for x in range(len(r_oc_sift)):
       r_oc_sift[x], matches_mask_oc_sift[x] =  ratio_test_flann(sift_permutations_outside_chair[x])

print(f'sift outside chair cup permutations: {len(r_oc_sift[0])},  {len(r_oc_sift[1])}, {len(r_oc_sift[2])}') 
print(f'sift outside chair plate permutations: {len(r_oc_sift[3])}, {len(r_oc_sift[4])}, {len(r_oc_sift[5])}') 
print(f'sift outside chair bottle permutations: {len(r_oc_sift[6])}, {len(r_oc_sift[7])}, {len(r_oc_sift[8])}')

'''
plot_flann(sift_permutations_inside_desk[0], matches_mask_id_sift[0], inside_cup, sift_kpi1, inside_robot_desk_cup, sift_kpi4)
plot_flann(sift_permutations_inside_desk[1], matches_mask_id_sift[1], inside_cup, sift_kpi1, inside_robot_desk_plate, sift_kpi5)
plot_flann(sift_permutations_inside_desk[2], matches_mask_id_sift[2], inside_cup, sift_kpi1, inside_robot_desk_bottle, sift_kpi6)
plot_flann(sift_permutations_inside_desk[3], matches_mask_id_sift[3], inside_plate, sift_kpi2, inside_robot_desk_cup, sift_kpi4)
plot_flann(sift_permutations_inside_desk[4], matches_mask_id_sift[4], inside_plate, sift_kpi2, inside_robot_desk_plate, sift_kpi5)
plot_flann(sift_permutations_inside_desk[5], matches_mask_id_sift[5], inside_plate, sift_kpi2, inside_robot_desk_bottle, sift_kpi6)
plot_flann(sift_permutations_inside_desk[6], matches_mask_id_sift[6], inside_bottle, sift_kpi3, inside_robot_desk_cup, sift_kpi4)
plot_flann(sift_permutations_inside_desk[7], matches_mask_id_sift[7], inside_bottle, sift_kpi3, inside_robot_desk_plate, sift_kpi5)
plot_flann(sift_permutations_inside_desk[8], matches_mask_id_sift[8], inside_bottle, sift_kpi3, inside_robot_desk_bottle, sift_kpi6)
'''
