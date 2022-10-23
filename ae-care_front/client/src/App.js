import React from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Nav1 from './components/Nav';
import Main from './components/Main';
import Weekly from './components/Weekly';
import Monthly from './components/Monthly';
import Videos from './components/Videos';
import Footer from './components/Footer';


function App() {
  return (
    <div>
      <Router>
        <Nav1/>
        <Routes>
          <Route path="/component/Main" element={<Main />} />
          <Route path="/component/Weekly" element={<Weekly />} />
          <Route path="/component/Monthly" element={<Monthly />} />
          <Route path="/component/Videos" element={<Videos />} />
        </Routes>
        <Footer/>
      </Router>
    </div>
  );
}



export default App;