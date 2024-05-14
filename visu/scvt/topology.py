import numpy as np
from scvt.timer import Timer


def update_dict(D, ii, jj, kk):
    D[ii] = [(jj, kk)] if ii  not in D else D[ii]+[(jj, kk)]
    D[jj] = [(ii, kk)] if jj  not in D else D[jj]+[(ii, kk)]
    D[kk] = [(jj, ii)] if kk  not in D else D[kk]+[(jj, ii)]





########################################################        

def simplex(D, ii,jj,kk):
    if ii>jj: ii,jj = jj,ii
    if jj>kk: jj,kk = kk,jj
    if ii>jj: ii,jj = jj,ii
    # now ii<jj<kk
    
    update_dict(D,ii,jj,kk)
    return (ii,jj,kk)


def enum_edges(simplices):
    for ii,jj,kk in simplices:
        # ii<jj<kk
        yield (ii,jj)
        yield (jj,kk)
        yield (ii,kk)

def insert_edge(edges, regions, edge,kk,n):
    if edge in edges:
        edges[edge]=(edges[edge], (n,kk))
    else:
        edges[edge]=(n,kk)
    ii,jj = edge
    if ii in regions:
        regions[ii].add(edge)
    else:
        regions[ii]={edge}
    if jj in regions:
        regions[jj].add(edge)
    else:
        regions[jj]={edge}

def all_edges_regions(simplices):
    # returns edges, regions
    # edge = ordered pair of generators
    # edges is a dictionary edge -> ((n1,point1, (n2,point2))
    # where n is the index of a simplex containing edge
    # and point is the index of the third point of that simplex
    # regions is a dictionary : point -> dict of edges 
    edges,regions = {},{}
    for n,(ii,jj,kk) in enumerate(simplices):
        # note that ii<jj<kk
        # edges (a,b) must have a<b
        insert_edge(edges,regions, (ii,jj),kk,n)
        insert_edge(edges,regions, (jj,kk),ii,n)
        insert_edge(edges,regions, (ii,kk),jj,n)
    return edges, [ regions[i] for i in range(len(regions)) ]

def dual_edges(edges):
    for (ii,jj),((n1,kk1),(n2,kk2)) in edges.items():
        yield (n1,n2)        

def regions_vertices(regions, edges):
    for region in regions:
        for edge in region:
            (n1,p1), (n2,p2) = edges[edge]
            yield n1
            yield n2
