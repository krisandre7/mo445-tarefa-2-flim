using namespace std;
#include <math.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <set>
#include <assert.h>
#include "image.h"
#include "geometry.h"

class Ellipse {
public:
  int x;
  int y;
  double a;
  double b;
  double sin_teta;
  double cos_teta;
  double weight;
  Ellipse(int _x, int _y, double _a, double _b,
	  double _sin_teta, double _cos_teta,
	  double _weight): x(_x), y(_y), a(_a), b(_b),
			 sin_teta(_sin_teta),
			 cos_teta(_cos_teta),
			 weight(_weight) { };
  inline bool operator < (Ellipse e)   {
    return (weight > e.weight);
  }
};


void FillFrame(Image *img, int val);

set< pair<int, int> > getEllipse(int xcen, int ycen,
				 double a, double b,
				 double sphi, double cphi);

/**
 * Retorna um vetor de Ellipses com todas as elipses encontradas
 * na imagem binária img, com restricão de centro de elipse à
 * máscara mask, que estejam de acordo com os parâmetros mínimos
 * e máximos fornecidos na entrada.
 * @param img Imagem binária de entrada.
 * @param mask Imagem binária usada como máscara para o centro das
 *             elipses.
 * @param grad Imagem gradiente.
 * @param m_pairs Número de retas de amostragem tracadas para
 *                cada ponto avaliado.
 * @param L tamanho máximo das elipses.
 * @param min_area Área mínima de uma elipse.
 * @param max_area Área máxima de uma elipse.
 * @param min_a Tamanho do eixo A mínimo de uma elipse.
 * @param max_a Tamanho do eixo A máximo de uma elipse.
 * @param min_b Tamanho do eixo A mínimo de uma elipse.
 * @param max_b Tamanho do eixo A máximo de uma elipse.
 * @param max_a_b Razão A/B máxima de uma elipse.
 * @param min_dist Distância mínima entre as elipses encontradas.
 * @return Vetor de objetos Ellipse com todas as elipses encontradas.
 **/
vector<Ellipse> findEllipses(Image *img, Image *mask, Image *grad,
			       unsigned int m_pairs,
			       unsigned int L,
			       unsigned int min_area,
			       unsigned int max_area,
			       double min_a,   double max_a,
			       double min_b,   double max_b,
			       double max_a_b,
			       unsigned int min_dist) {

  vector<Ellipse> ellipses, ellipses2;
  double x, y, aux;
  int p;
  unsigned int i, j, k, v;
  unsigned int L2 = L * L;
  unsigned int ncols = img->ncols;
  unsigned int nrows = img->nrows;

  Image *bin2 = ift_CopyImage(img);
  FillFrame(bin2, 0);

  //---------------------------------------------------

  Vector *tau    = (Vector *) malloc(sizeof(Vector) * m_pairs);
  Point *epsilon = (Point *) malloc(sizeof(Point) * m_pairs);

  double teta = 0.0;
  for (i = 0; i < m_pairs; i++) {
    tau[i].x = cosf(teta);
    tau[i].y = sinf(teta);
    tau[i].z = 0.0;
    teta += (M_PI / m_pairs);
  }

  for (i = 1; i < nrows - 1; i++) {
    for (j = 1; j < ncols - 1; j++) {

      p = bin2->tbrow[i] + j;
      if ((mask->val[p] == 0) || (bin2->val[p] == 0))
        continue;

      //--------- Sample lines ---------------------------
      double gSxy = 0;
      double gSy2 = 0;
      double gSx2 = 0;
      double xc = j + 0.5;
      double yc = i + 0.5;
      for (k = 0; k < m_pairs; k++) {
        x = tau[k].x;
        y = tau[k].y;
	int grad_max = 0;
	double grad_max_x = 0;
	double grad_max_y = 0;

	/* encontra o tamanho maximo do raio no maior gradiente */
        for (v = 1; v < L; v++) {
          int q1 = (int)(xc + x) + bin2->tbrow[(int)(yc + y)];
          if (bin2->val[q1] == 0)
            break;

          int q2 = (int)(xc - x) + bin2->tbrow[(int)(yc - y)];
          if (bin2->val[q2] == 0)
            break;

          x += tau[k].x;
          y += tau[k].y;

	  if (grad->val[q1] >= grad_max) {
	    grad_max = grad->val[q1];
	    grad_max_x = x;
	    grad_max_y = y;
	  }
	  if (grad->val[q2] >= grad_max) {
	    grad_max = grad->val[q2];
	    grad_max_x = x;
	    grad_max_y = y;
	  }
        }

	x = grad_max_x;
	y = grad_max_y;

        epsilon[k].x = x;
        epsilon[k].y = -y;

        gSxy -= x*y; // gSxy += x*(-y);
        gSx2 += x*x;
        gSy2 += y*y;
      }

      //-------------------- TETA --------------------------

      double sum     = gSx2 + gSy2;
      double Delta   = sqrt(sum * sum - 4.0 * (gSx2 * gSy2 - gSxy * gSxy));
      double lambda1 = (sum + Delta) / 2.0;
      double eigx, eigy;

      if (gSxy == 0.0) {
        if (gSx2 > gSy2) {
          eigy = 0.0;
          eigx = 1.0;
        } else {
          eigy = 1.0;
          eigx = 0.0;
        }
      } else {
        eigy = 1.0;
        eigx = (lambda1 - gSy2) / gSxy;
        double norm = sqrt(eigx * eigx + eigy * eigy);
        eigy /= norm;
        eigx /= norm;
      }

      if (eigx == 0.0)
        teta = - M_PI_2;
      else
        teta = atan(-eigy / eigx);

      //----------------- A & B ---------------------------------

      if (teta < 0.0)
        teta += M_PI;
      if (teta > M_PI)
        teta = M_PI;
      teta = M_PI - teta;

      double sin_teta = sin(teta);
      double cos_teta = cos(teta);
      double Su2v2 = 0;
      double Su4 = 0;
      double Sv4 = 0;
      double Su2 = 0;
      double Sv2 = 0;
      double uu, vv;
      for (k = 0; k < m_pairs; k++) {
        uu = epsilon[k].x;
        vv = epsilon[k].y;
        aux = uu * cos_teta - vv * sin_teta;
        vv = vv * cos_teta + uu * sin_teta;
        uu = aux * aux;
        vv *= vv;
        Su2v2 += uu * vv;
        Su2 += uu;
        Sv2 += vv;
        Su4 += uu * uu;
        Sv4 += vv * vv;
      }

      aux = Su2v2 * Su2v2 - Su4 * Sv4;
      double a2 = (aux / (Su2v2 * Sv2 - Su2 * Sv4));
      double b2 = (aux / (Su2 * Su2v2 - Su4 * Sv2));

      if (isnan(a2) || a2 < 0.0)
        a2 = 1.0;
      if (isnan(b2) || b2 < 0.0)
        b2 = 1.0;
      if (a2 > L2)
        a2 = L2;
      if (b2 > L2)
        b2 = L2;

      double l1E, l2E;
      if (a2 >= b2) {
        l1E = a2;
        l2E = b2;
      } else {
        l1E = b2;
        l2E = a2;
      }

      double a = sqrt(l1E);
      double b = sqrt(l2E);
      double area = M_PI * a * b;
      double a_b = a / b;

      /* ignora a elipse caso esteja fora dos parametros de entrada */
      if ((a < min_a) || (b < min_b) || (a > max_a) || (b > max_b)
	  || (a_b > max_a_b)
	  || (area < min_area) || (area > max_area)) {
	continue;
      }

      /* determine ellipse cost */
      set< pair<int, int> > ellipse = getEllipse(j, i, a, b,
						 sin_teta, cos_teta);
      set< pair<int, int> >::iterator it;
      double weight = 0;
      for (it = ellipse.begin(); it != ellipse.end(); it++) {
	int ex = (*it).first;
	int ey = (*it).second;
	if ((ex >= 0) && (ex < grad->ncols) && (ey >= 0) && (ey < grad->nrows)) {
	  weight += grad->val[ex + grad->tbrow[ey]];
	}
      }
      weight /= ellipse.size();

      ellipses.push_back(Ellipse(j, i, a, b, sin_teta, cos_teta, weight));
      
    } /* end for j */
  } /* end for i */

  /* sort the ellipses by decreasing weight */
  sort (ellipses.begin(), ellipses.end());

  /* remove ellipses that are too close */
  size_t n_ellipses = ellipses.size();
  bool ellipse_ok[n_ellipses];
  for (i = 0; i < n_ellipses; i++)
    ellipse_ok[i] = true;
  int min_dist2 = min_dist * min_dist;
  for (i = 0; i < n_ellipses; i++) {
    if (ellipse_ok[i]) {
      ellipses2.push_back(ellipses[i]);
      for (j = i + 1; j < n_ellipses; j++) {
	int dx = ellipses[j].x - ellipses[i].x;
	int dy = ellipses[j].y - ellipses[i].y;
	if (dx * dx + dy * dy < min_dist2)
	  ellipse_ok[j] = false; // too close to another with greater weight
      }
    }
  }

  free(tau);
  free(epsilon);
  DestroyImage(&bin2);

  return ellipses2;
}

/**
 * Get a set with the contour of an ellipse.
 * @param xcen Center of ellipse X.
 * @param ycen Center of ellipse Y;
 * @param a Radius of ellipse A;
 * @param b Radius of ellipse B;
 * @param sphi Sine of rotation angle;
 * @param cphi Cosine of rotation angle;
 * @return Set of pair<int,int> with the contour of the given ellipse.
 **/
set< pair<int, int> > getEllipse(int xcen, int ycen,
				 double a, double b,
				 double sphi, double cphi) {

  pair<int, int> p[4]; /**< current iteration points */
  pair<int, int> pp[4]; /**< last iteration points */
  set< pair<int, int> > ellipse;
  
  double cphisqr = cphi * cphi;
  double sphisqr = sphi * sphi;
  double asqr = a * a;
  double bsqr = b * b;
  
  double c1 = (cphisqr / asqr) + (sphisqr / bsqr);
  double c2 = ((cphi * sphi / asqr) - (cphi * sphi / bsqr)) / c1;
  double c3 = (bsqr * cphisqr) + (asqr * sphisqr);
  int ymax = (int)(sqrt(c3) + 0.5);
  double c4 = a * b / c3;
  double c5 = 0;
  double v1 = c4 * c4;
  double c6 = 2 * v1;
  c3 = c3 * v1 - v1;

  unsigned int i = 0;
  int yy = 0;
  double d = 0;

  /* odd first points */
  if (ymax % 2) {
    d = sqrt(c3);
    ellipse.insert(pair<int, int> ((int)(xcen - d + 0.5), ycen));
    ellipse.insert(pair<int, int> ((int)(xcen + d + 0.5), ycen));
    c5 = c2;
    yy = 1;
  }
  bool first = true;
  while (c3 >= 0) {
    d = sqrt(c3);
    double xleft = c5 - d;
    double xright = c5 + d;
    p[0].first = (int)(xcen + xleft + 0.5);
    p[0].second = ycen + yy;
    p[1].first = (int)(xcen + xright + 0.5);
    p[1].second = p[0].second;
    p[2].first = (int)(xcen - xright + 0.5);
    p[2].second = ycen - yy;
    p[3].first = (int)(xcen - xleft + 0.5);
    p[3].second = p[2].second;

    if (first) {
      memcpy(pp, p, sizeof(p));
      first = false;
    }

    for (i = 0; i < 4; i++) {

      if ((abs(p[i].first - pp[i].first) > 1)
	|| (abs(p[i].second - pp[i].second) > 1)) {

	int x0 = p[i].first;
	int y0 = p[i].second;
	int dx, dy, xinc, yinc, res1, res2;
	
	xinc = 1; yinc = 1;
	dx = pp[i].first - x0;
	if (dx < 0) {
	  xinc = -1;
	  dx = -dx;
	}
	dy = pp[i].second - y0;
	if (dy < 0) {
	  yinc = -1;
	  dy = -dy;
	}
	res1 = 0;
	res2 = 0;

	if (dx > dy) {
	  while (x0 != pp[i].first) {
	    ellipse.insert(pair<int, int> (x0, y0));
	    if (res1 > res2) {
	      res2 += dx - res1;
	      res1 = 0;
	      y0 += yinc;
	    }
	    res1 += dy;
	    x0 += xinc;
	  }
	}
	else if (dx < dy) {
	  while (y0 != pp[i].second) {
	    ellipse.insert(pair<int, int> (x0, y0));
	    if (res1 > res2) {
	      res2 += dy - res1;
	      res1 = 0;
	      x0 += xinc;
	    }
	    res1 += dx;
	    y0 += yinc;
	  }
	}
	else {
	  while (x0 != pp[i].first) {
	    ellipse.insert(pair<int, int> (x0, y0));
	    y0 += yinc;
	    x0 += xinc;
	  }
	}
      } else
	ellipse.insert(p[i]);
    }

    memcpy(pp, p, sizeof(p));

    c5 += c2;
    v1 += c6;
    c3 -= v1;
    yy += 1;
  }

  int x;
  for (x = p[0].first; x < p[1].first; x++)
    ellipse.insert(pair<int, int> (x, p[0].second));
  for (x = p[2].first; x < p[3].first; x++)
    ellipse.insert(pair<int, int> (x, p[2].second));

  return ellipse;

}

void FillFrame(Image *img, int val) {
  int i, p;
  p = img->tbrow[img->nrows - 1];
  for (i = 0; i < img->ncols; ++i) {
    img->val[i] = val; /* pinta a primeira linha */
    img->val[i + p] = val; /* pinta a ultima linha */
  }
  for (i = 0; i < img->nrows; ++i) {
    p = img->tbrow[i];
    img->val[p] = val; /* pinta a primeira coluna */
    img->val[img->ncols - 1 + p] = val; /* pinta a ultima coluna */
  }
}


