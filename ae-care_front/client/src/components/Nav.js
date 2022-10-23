import React, { Component } from 'react';
import { Container, Nav, Navbar } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import { MDBBtn } from 'mdb-react-ui-kit';

class Nav1 extends Component {
    render() {
        return (
            <div>
                <Navbar collapseOnSelect expand="lg" bg="dark" variant="dark">
                    <Container>
                    <Navbar.Brand><h1>ae-Care</h1></Navbar.Brand>
                    <Navbar.Toggle aria-controls="responsive-navbar-nav" />
                    <Navbar.Collapse id="responsive-navbar-nav">
                        <Nav className="me-auto">
                            <ul>
                                <Link to="/component/Main">
                                    <MDBBtn outline rounded>Main</MDBBtn>
                                </Link>
                            </ul>
                            <ul>
                                <Link to="/component/Weekly">
                                    <MDBBtn outline rounded>Weekly</MDBBtn>
                                </Link>  
                            </ul>
                            <ul>
                                <Link to="/component/Monthly">
                                    <MDBBtn outline rounded>Monthly</MDBBtn>
                                </Link>
                            </ul>
                            <ul>
                                <Link to="/component/Videos">
                                    <MDBBtn outline rounded>Videos</MDBBtn>
                                </Link>
                            </ul>
                        </Nav>
                    </Navbar.Collapse>
                    </Container>
                </Navbar>
            </div>
        );
    }
}

export default Nav1;