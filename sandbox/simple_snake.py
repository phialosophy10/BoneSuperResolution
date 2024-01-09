import numpy as np
import scipy.interpolate
import scipy.linalg
import skimage.draw


def make_circular_snake(N, center, radius):
    """ Initialize circular snake as a 2-by-N array."""
    center = center.reshape([2,1])
    angles = np.arange(N)*2*np.pi/N
    return(center+radius*np.array([np.cos(angles), np.sin(angles)]))


def normalize(n):
    return n/np.sqrt(np.sum(n**2,axis=0))


def snake_normals(snake):
    """ Returns snake normals. Expects snake to be 2-by-N array."""
    ds = normalize(np.roll(snake, 1, axis=1) - snake) 
    tangent = normalize(np.roll(ds,-1,axis=1) + ds)
    normal = tangent[[1,0],:]*np.array([-1,1]).reshape([2,1])
    return(normal)


def distribute_points(snake):
    """ Distributes snake points equidistantly. Expects snake to be 2-by-N array."""
    N = snake.shape[1]
    d = np.sqrt(np.sum((np.roll(snake, -1, axis=1)-snake)**2, axis=0)) # length of line segments
    f = scipy.interpolate.interp1d(np.r_[0, np.cumsum(d)], np.c_[snake, snake[:,0:1]])
    return(f(sum(d)*np.arange(N)/N))


def is_crossing(p1, p2, p3, p4):
    """ Check if the line segments (p1, p2) and (p3, p4) cross."""
    crossing = False
    d21 = p2 - p1
    d43 = p4 - p3
    d31 = p3 - p1
    det = d21[0]*d43[1] - d21[1]*d43[0] # Determinant
    if det != 0.0 and d21[0] != 0.0 and d21[1] != 0.0:
        a = d43[0]/d21[0] - d43[1]/d21[1]
        b = d31[1]/d21[1] - d31[0]/d21[0]
        if a != 0.0:
            u = b/a
            if d21[0] > 0:
                t = (d43[0]*u + d31[0])/d21[0]
            else:
                t = (d43[1]*u + d31[1])/d21[1]
            crossing = 0 < u < 1 and 0 < t < 1         
    return crossing


def is_counterclockwise(snake):
    """ Check if points are ordered counterclockwise."""
    return np.dot(snake[0,1:] - snake[0,:-1],
                  snake[1,1:] + snake[1,:-1]) < 0


def remove_intersections(snake):
    """ Reorder snake points to remove self-intersections.
        Arguments: snake represented by a 2-by-N array.
        Returns: snake.
    """
    pad_snake = np.append(snake, snake[:,0].reshape(2,1), axis=1)
    pad_n = pad_snake.shape[1]
    n = pad_n - 1 
    
    for i in range(pad_n - 3):
        for j in range(i + 2, pad_n - 1):
            pts = pad_snake[:,[i, i + 1, j, j + 1]]
            if is_crossing(pts[:,0], pts[:,1], pts[:,2], pts[:,3]):
                # Reverse vertices of smallest loop
                rb = i + 1 # Reverse begin
                re = j     # Reverse end
                if j - i > n // 2:
                    # Other loop is smallest
                    rb = j + 1
                    re = i + n                    
                while rb < re:
                    ia = rb % n
                    rb = rb + 1                    
                    ib = re % n
                    re = re - 1                    
                    pad_snake[:,[ia, ib]] = pad_snake[:,[ib, ia]]                    
                pad_snake[:,-1] = pad_snake[:,0]                
    snake = pad_snake[:,:-1]
    if is_counterclockwise(snake):
        return snake
    else:
        return np.flip(snake, axis=1)
    
    
def keep_snake_inside(snake, shape):
    """ Contains snake inside the image."""
    snake[snake<0]=0
    snake[0][snake[0]>shape[0]-1] = shape[0]-1 
    snake[1][snake[1]>shape[1]-1] = shape[1]-1 
    return snake

    
def regularization_matrix(N, alpha, beta):
    """ Matrix for smoothing the snake."""
    d = alpha*np.array([-2, 1, 0, 0]) + beta*np.array([-6, 4, -1, 0])
    D = np.fromfunction(lambda i,j: np.minimum((i-j)%N,(j-i)%N), (N,N), dtype=np.int)
    A = d[np.minimum(D,len(d)-1)]
    return(scipy.linalg.inv(np.eye(N)-A))


def evolve_snake(snake, I, B, step_size):
    """ Single step of snake evolution."""
    mask = skimage.draw.polygon2mask(I.shape, snake.T)
    m_in = np.mean(I[mask])
    m_out = np.mean(I[~mask])
      
    f = scipy.interpolate.RectBivariateSpline(np.arange(I.shape[0]), np.arange(I.shape[1]), I)
    val = f(snake[0],snake[1], grid=False)
    # val = I[snake[0].astype(int), snake[1].astype(int)] # simpler variant without interpolation
    force = 0.5*(m_in-m_out)*(2*val - (m_in+m_out))
    snake += step_size*force*snake_normals(snake) # external part
    snake = np.dot(snake, B) # internal part, ordering influenced by 2-by-N representation of snake
    snake = remove_intersections(snake)
    snake = distribute_points(snake)
    snake = keep_snake_inside(snake, I.shape)
    return snake