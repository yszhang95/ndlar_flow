#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py


# In[2]:


import matplotlib
import matplotlib.pyplot as plt


# In[3]:


import numpy as np


# In[4]:


f = h5py.File('/home/yousen/Public/ndlar_shared/data/packet-0050015-2024_07_08_13_37_49_CDT.FLOW_evt7.hdf5')


# In[5]:


selected = f['hits/selected/data']
deselected = f['hits/deselected/data']


# In[6]:


src = f['source_file/data'][0].decode('utf-8')


# In[7]:


fsrc = h5py.File(src)


# In[8]:


fsrc.keys()


# In[9]:


fsrc['charge'].keys()


# In[10]:


fsrc['/charge/events/ref'].keys()


# In[11]:


eventmask = fsrc['/charge/events/ref/charge/calib_prompt_hits/ref'][:,0] == 7


# In[12]:


hits = fsrc['/charge/calib_prompt_hits/data'][eventmask]


# In[13]:


len(hits)


# In[14]:


len(selected) + len(deselected)


# In[15]:


def plot_proj(hits_dict):
    nrows = 3
    ncols = 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows))
    for k, v in hits_dict.items():
        axs[0,0].scatter(x=v['x'], y=v['y'], marker='o', label=k)
        axs[0,0].set_xlabel('x')
        axs[0,0].set_ylabel('y')
        axs[0,1].scatter(x=v['x'], y=v['z'], marker='o', label=k)
        axs[0,1].set_xlabel('x')
        axs[0,1].set_ylabel('z')
        axs[1,0].scatter(x=v['y'], y=v['z'], marker='o', label=k)
        axs[1,0].set_xlabel('y')
        axs[1,0].set_ylabel('z')
        axs[1,1].scatter(x=v['x'], y=v['ts_pps'], marker='o', label=k)
        axs[1,1].set_xlabel('x')
        axs[1,1].set_ylabel('ts_pps')
        axs[2,0].scatter(x=v['x'], y=v['t_drift'], marker='o', label=k)
        axs[2,0].set_xlabel('x')
        axs[2,0].set_ylabel('t_drift')
        axs[2,1].scatter(x=v['Q'], y=v['t_drift'], marker='o', label=k)
        axs[2,1].set_xlabel('Q')
        axs[2,1].set_ylabel('t_drift')
    for i in range(nrows):
        for j in range(ncols):
            axs[i,j].legend()
    return fig, axs


# In[21]:


plt.ioff()
fig3, _ = plot_proj({'all' : hits, 'selected' : selected, 'deselected' : deselected})
fig2, _ = plot_proj({'all' : hits, 'selected' : selected})
fig1, _ = plot_proj({'all' : hits})
fig3.savefig('test_all_selected_deselected.jpg')
fig2.savefig('test_all_selected.jpg')
fig1.savefig('test_all.jpg')


# In[18]:


for h in selected:
    pt = np.array([h['x'], h['y'], h['z'], h['Q'], h['t_drift'], h['ts_pps']]).reshape(1, -1)
    pts = np.column_stack([hits['x'], hits['y'], hits['z'], hits['Q'], hits['t_drift'], hits['ts_pps']])
    d = np.linalg.norm(pts-pt, axis=1)
    found = np.sum(d<1E-6)
    if not found:
        print('faied', pt)


# In[19]:


for h in deselected:
    pt = np.array([h['x'], h['y'], h['z'], h['Q'], h['t_drift'], h['ts_pps']]).reshape(1, -1)
    pts = np.column_stack([hits['x'], hits['y'], hits['z'], hits['Q'], hits['t_drift'], hits['ts_pps']])
    d = np.linalg.norm(pts-pt, axis=1)
    found = np.sum(d<1E-6)
    if not found:
        print('faied', pt)


# In[ ]:




